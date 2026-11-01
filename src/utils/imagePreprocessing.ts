import { loadImage } from 'react-native-nitro-image';

export const MODEL_IMAGE_SIZE = 416;

export interface ImageTransform {
  sourceWidth: number;
  sourceHeight: number;
  scaleX: number;
  scaleY: number;
  padLeft: number;
  padTop: number;
}

export interface PreprocessedImage {
  input: ArrayBuffer;
  transform: ImageTransform;
}

const CHANNEL_OFFSETS: Record<string, [number, number, number]> = {
  ARGB: [1, 2, 3],
  BGRA: [2, 1, 0],
  ABGR: [3, 2, 1],
  RGBA: [0, 1, 2],
  XRGB: [1, 2, 3],
  BGRX: [2, 1, 0],
  XBGR: [3, 2, 1],
  RGBX: [0, 1, 2],
  RGB: [0, 1, 2],
  BGR: [2, 1, 0],
};

function getImageFilePath(uri: string): string {
  if (!uri.startsWith('file://')) {
    throw new Error('The captured photo does not have a local file URI.');
  }

  return decodeURIComponent(uri.slice('file://'.length));
}

export async function preprocessImage(uri: string): Promise<PreprocessedImage> {
  let sourceImage: Awaited<ReturnType<typeof loadImage>> | undefined;
  let resizedImage: Awaited<ReturnType<typeof loadImage>> | undefined;

  try {
    sourceImage = await loadImage({ filePath: getImageFilePath(uri) });
    const sourceWidth = sourceImage.width;
    const sourceHeight = sourceImage.height;

    if (
      !Number.isFinite(sourceWidth) ||
      !Number.isFinite(sourceHeight) ||
      sourceWidth <= 0 ||
      sourceHeight <= 0
    ) {
      throw new Error('The captured photo has invalid dimensions.');
    }

    const scale = Math.min(
      MODEL_IMAGE_SIZE / sourceWidth,
      MODEL_IMAGE_SIZE / sourceHeight
    );
    const resizedWidth = Math.max(1, Math.round(sourceWidth * scale));
    const resizedHeight = Math.max(1, Math.round(sourceHeight * scale));
    const padLeft = Math.floor((MODEL_IMAGE_SIZE - resizedWidth) / 2);
    const padTop = Math.floor((MODEL_IMAGE_SIZE - resizedHeight) / 2);

    resizedImage = await sourceImage.resizeAsync(resizedWidth, resizedHeight);
    const pixels = await resizedImage.toRawPixelDataAsync(false);
    const channelOffsets = CHANNEL_OFFSETS[pixels.pixelFormat];

    if (!channelOffsets) {
      throw new Error(`Unsupported photo pixel format: ${pixels.pixelFormat}`);
    }

    const bytesPerPixel = pixels.pixelFormat === 'RGB' || pixels.pixelFormat === 'BGR' ? 3 : 4;
    const expectedPixelBytes = resizedWidth * resizedHeight * bytesPerPixel;

    if (
      pixels.width !== resizedWidth ||
      pixels.height !== resizedHeight ||
      pixels.buffer.byteLength !== expectedPixelBytes
    ) {
      throw new Error('The decoded photo has an invalid pixel buffer.');
    }

    const planeSize = MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE;
    const input = new ArrayBuffer(planeSize * 3 * Float32Array.BYTES_PER_ELEMENT);
    const tensor = new Float32Array(input);
    const paddingValue = 114 / 255;
    tensor.fill(paddingValue);

    const sourcePixels = new Uint8Array(pixels.buffer);
    const rowLength = resizedWidth * bytesPerPixel;

    for (let y = 0; y < resizedHeight; y += 1) {
      for (let x = 0; x < resizedWidth; x += 1) {
        const sourceOffset = y * rowLength + x * bytesPerPixel;
        const targetOffset =
          (y + padTop) * MODEL_IMAGE_SIZE + x + padLeft;

        tensor[targetOffset] = sourcePixels[sourceOffset + channelOffsets[0]] / 255;
        tensor[planeSize + targetOffset] =
          sourcePixels[sourceOffset + channelOffsets[1]] / 255;
        tensor[planeSize * 2 + targetOffset] =
          sourcePixels[sourceOffset + channelOffsets[2]] / 255;
      }
    }

    return {
      input,
      transform: {
        sourceWidth,
        sourceHeight,
        scaleX: resizedWidth / sourceWidth,
        scaleY: resizedHeight / sourceHeight,
        padLeft,
        padTop,
      },
    };
  } finally {
    resizedImage?.dispose();
    sourceImage?.dispose();
  }
}
