import { MODEL_IMAGE_SIZE, type ImageTransform } from './imagePreprocessing';

export const CLASS_NAMES = [
  'almond',
  'apple',
  'asparagus',
  'avocado',
  'banana',
  'beans',
  'beet',
  'bell pepper',
  'blackberry',
  'blueberry',
  'broccoli',
  'brussels sprouts',
  'cabbage',
  'carrot',
  'cauliflower',
  'celery',
  'cherry',
  'corn',
  'cucumber',
  'egg',
  'eggplant',
  'garlic',
  'grape',
  'green bean',
  'green onion',
  'hot pepper',
  'kiwi',
  'lemon',
  'lettuce',
  'lime',
  'mandarin',
  'mushroom',
  'onion',
  'orange',
  'pattypan squash',
  'pea',
  'peach',
  'pear',
  'pineapple',
  'potato',
  'pumpkin',
  'radish',
  'raspberry',
  'strawberry',
  'tomato',
  'vegetable marrow',
  'watermelon',
] as const;

export const MODEL_OUTPUT_SHAPE = [1, 51, 3549] as const;
export const CONFIDENCE_THRESHOLD = 0.25;
export const NMS_IOU_THRESHOLD = 0.45;

export interface Detection {
  classId: number;
  name: (typeof CLASS_NAMES)[number];
  confidence: number;
  left: number;
  top: number;
  right: number;
  bottom: number;
}

function intersectionOverUnion(first: Detection, second: Detection): number {
  const left = Math.max(first.left, second.left);
  const top = Math.max(first.top, second.top);
  const right = Math.min(first.right, second.right);
  const bottom = Math.min(first.bottom, second.bottom);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const firstArea =
    (first.right - first.left) * (first.bottom - first.top);
  const secondArea =
    (second.right - second.left) * (second.bottom - second.top);
  const union = firstArea + secondArea - intersection;

  return union > 0 ? intersection / union : 0;
}

export function decodeYoloOutput(
  output: Float32Array,
  transform: ImageTransform,
  confidenceThreshold = CONFIDENCE_THRESHOLD,
  iouThreshold = NMS_IOU_THRESHOLD
): Detection[] {
  const anchorCount = MODEL_OUTPUT_SHAPE[2];
  const channelCount = MODEL_OUTPUT_SHAPE[1];

  if (output.length !== channelCount * anchorCount) {
    throw new Error(
      `Invalid model output length: expected ${channelCount * anchorCount}, received ${output.length}.`
    );
  }

  const candidates: Detection[] = [];

  for (let anchor = 0; anchor < anchorCount; anchor += 1) {
    let classId = -1;
    let confidence = Number.NEGATIVE_INFINITY;

    for (let classIndex = 0; classIndex < CLASS_NAMES.length; classIndex += 1) {
      const score = output[(4 + classIndex) * anchorCount + anchor];
      if (Number.isFinite(score) && score > confidence) {
        confidence = score;
        classId = classIndex;
      }
    }

    if (classId < 0 || confidence < confidenceThreshold) {
      continue;
    }

    const centerX =
      output[anchor] * MODEL_IMAGE_SIZE;
    const centerY =
      output[anchorCount + anchor] * MODEL_IMAGE_SIZE;
    const boxWidth =
      output[anchorCount * 2 + anchor] * MODEL_IMAGE_SIZE;
    const boxHeight =
      output[anchorCount * 3 + anchor] * MODEL_IMAGE_SIZE;

    if (
      !Number.isFinite(centerX) ||
      !Number.isFinite(centerY) ||
      !Number.isFinite(boxWidth) ||
      !Number.isFinite(boxHeight) ||
      boxWidth <= 0 ||
      boxHeight <= 0
    ) {
      continue;
    }

    const left = Math.max(
      0,
      Math.min(
        transform.sourceWidth,
        (centerX - boxWidth / 2 - transform.padLeft) / transform.scaleX
      )
    );
    const top = Math.max(
      0,
      Math.min(
        transform.sourceHeight,
        (centerY - boxHeight / 2 - transform.padTop) / transform.scaleY
      )
    );
    const right = Math.max(
      0,
      Math.min(
        transform.sourceWidth,
        (centerX + boxWidth / 2 - transform.padLeft) / transform.scaleX
      )
    );
    const bottom = Math.max(
      0,
      Math.min(
        transform.sourceHeight,
        (centerY + boxHeight / 2 - transform.padTop) / transform.scaleY
      )
    );

    if (right <= left || bottom <= top) {
      continue;
    }

    candidates.push({
      classId,
      name: CLASS_NAMES[classId],
      confidence,
      left,
      top,
      right,
      bottom,
    });
  }

  candidates.sort((first, second) => second.confidence - first.confidence);
  const detections: Detection[] = [];

  for (const candidate of candidates) {
    const overlapsExistingDetection = detections.some(
      (detection) =>
        detection.classId === candidate.classId &&
        intersectionOverUnion(detection, candidate) > iouThreshold
    );

    if (!overlapsExistingDetection) {
      detections.push(candidate);
    }
  }

  return detections;
}
