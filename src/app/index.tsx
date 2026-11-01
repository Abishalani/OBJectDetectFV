import { CameraView, useCameraPermissions } from 'expo-camera';
import type { CameraView as CameraViewType } from 'expo-camera';
import { loadTensorflowModel } from 'react-native-fast-tflite';
import type { TensorflowModel } from 'react-native-fast-tflite';
import { useEffect, useRef, useState } from 'react';
import {
  Button,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  MODEL_IMAGE_SIZE,
  preprocessImage,
} from '../utils/imagePreprocessing';
import {
  decodeYoloOutput,
  MODEL_OUTPUT_SHAPE,
  type Detection,
} from '../utils/yoloPostprocessing';

type ModelState = 'loading' | 'ready' | 'error';
type DetectionState = 'idle' | 'processing' | 'done' | 'error';

// Metro needs a static require to bundle the TFLite asset.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const tfliteModelAsset = require('../../assets/best.tflite');

interface PhotoSize {
  width: number;
  height: number;
}

function validateModel(model: TensorflowModel): void {
  const input = model.inputs[0];
  const output = model.outputs[0];
  const expectedInputShape = [1, 3, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE];

  if (
    model.inputs.length !== 1 ||
    input?.dataType !== 'float32' ||
    input.shape.join(',') !== expectedInputShape.join(',')
  ) {
    throw new Error('The model input must be float32 with shape [1, 3, 416, 416].');
  }

  if (
    model.outputs.length !== 1 ||
    output?.dataType !== 'float32' ||
    output.shape.join(',') !== MODEL_OUTPUT_SHAPE.join(',')
  ) {
    throw new Error('The model output must be float32 with shape [1, 51, 3549].');
  }
}

function toDisplayName(name: string): string {
  return name.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function HomeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraViewType | null>(null);
  const inferenceInProgress = useRef(false);
  const [cameraVisible, setCameraVisible] = useState(false);
  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [model, setModel] = useState<TensorflowModel | null>(null);
  const [modelState, setModelState] = useState<ModelState>('loading');
  const [modelStatus, setModelStatus] = useState('Loading detection model...');
  const [detectionState, setDetectionState] = useState<DetectionState>('idle');
  const [detectionMessage, setDetectionMessage] = useState('');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [photoSize, setPhotoSize] = useState<PhotoSize | null>(null);
  const [photoFrameSize, setPhotoFrameSize] = useState<PhotoSize>({
    width: 0,
    height: 0,
  });
  const [cameraError, setCameraError] = useState('');

  useEffect(() => {
    let isMounted = true;

    async function loadModel() {
      try {
        const loadedModel = await loadTensorflowModel(
          tfliteModelAsset,
          []
        );
        validateModel(loadedModel);
        console.log('47-class TFLite model loaded:', {
          inputs: loadedModel.inputs,
          outputs: loadedModel.outputs,
        });

        if (isMounted) {
          setModel(loadedModel);
          setModelState('ready');
          setModelStatus('Detection model ready');
        }
      } catch (error) {
        console.error('MODEL LOAD ERROR:', error);
        if (isMounted) {
          setModelState('error');
          setModelStatus('Could not load the detection model.');
        }
      }
    }

    void loadModel();
    return () => {
      isMounted = false;
    };
  }, []);

  const clearPhoto = () => {
    setPhotoUri(null);
    setPhotoSize(null);
    setDetections([]);
    setDetectionState('idle');
    setDetectionMessage('');
  };

  const processPhoto = async (
    uri: string,
    loadedModel: TensorflowModel
  ): Promise<void> => {
    let preprocessed: Awaited<ReturnType<typeof preprocessImage>>;

    try {
      preprocessed = await preprocessImage(uri);
      setPhotoSize({
        width: preprocessed.transform.sourceWidth,
        height: preprocessed.transform.sourceHeight,
      });
    } catch (error) {
      console.error('IMAGE PREPROCESSING ERROR:', error);
      setDetectionState('error');
      setDetectionMessage('Could not read this photo. Please take another photo.');
      return;
    }

    try {
      const outputBuffers = await loadedModel.run([preprocessed.input]);

      if (outputBuffers.length !== 1) {
        throw new Error(`Expected one output tensor, received ${outputBuffers.length}.`);
      }

      const outputBuffer = outputBuffers[0];
      if (!outputBuffer || outputBuffer.byteLength !== 51 * 3549 * Float32Array.BYTES_PER_ELEMENT) {
        throw new Error('The model returned an invalid output tensor.');
      }

      const output = new Float32Array(outputBuffer);
      const results = decodeYoloOutput(output, preprocessed.transform);

      console.log(
        'DETECTION RESULTS:',
        results.map(({ name, confidence, left, top, right, bottom }) => ({
          name,
          confidence,
          box: { left, top, right, bottom },
        }))
      );
      setDetections(results);
      setDetectionState('done');
      setDetectionMessage(
        results.length === 0
          ? 'No fruit or vegetable detected. Please try another photo.'
          : ''
      );
    } catch (error) {
      console.error('TFLITE INFERENCE OR OUTPUT ERROR:', error);
      setDetectionState('error');
      setDetectionMessage('Detection failed. Please try another photo.');
    }
  };

  const capturePhoto = async () => {
    if (inferenceInProgress.current) {
      return;
    }

    const camera = cameraRef.current;
    if (!camera) {
      setCameraError('The camera is not ready. Please try again.');
      return;
    }

    inferenceInProgress.current = true;
    try {
      setCameraError('');
      const photo = await camera.takePictureAsync();

      if (!photo?.uri) {
        throw new Error('The camera did not return a photo URI.');
      }

      setPhotoUri(photo.uri);
      setDetectionState('processing');
      setDetectionMessage('Detecting fruit or vegetable...');
      setDetections([]);
      setPhotoSize(null);

      if (!model) {
        setDetectionState('error');
        setDetectionMessage(
          modelState === 'error'
            ? 'The detection model could not be loaded.'
            : 'The detection model is still loading. Please try again shortly.'
        );
        return;
      }

      inferenceInProgress.current = true;
      await processPhoto(photo.uri, model);
    } catch (error) {
      console.error('CAMERA CAPTURE ERROR:', error);
      setCameraError('Could not capture this photo. Please try again.');
    } finally {
      inferenceInProgress.current = false;
    }
  };

  const closeScanner = () => {
    clearPhoto();
    setCameraError('');
    setCameraVisible(false);
  };

  const takeAnotherPhoto = () => {
    clearPhoto();
    setCameraError('');
  };

  const boxScale =
    photoSize && photoFrameSize.width > 0 && photoFrameSize.height > 0
      ? Math.min(
          photoFrameSize.width / photoSize.width,
          photoFrameSize.height / photoSize.height
        )
      : 0;
  const displayedPhotoWidth = photoSize ? photoSize.width * boxScale : 0;
  const displayedPhotoHeight = photoSize ? photoSize.height * boxScale : 0;
  const photoOffsetX = (photoFrameSize.width - displayedPhotoWidth) / 2;
  const photoOffsetY = (photoFrameSize.height - displayedPhotoHeight) / 2;

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.title}>NUTRILENS</Text>
        <Text style={styles.description}>
          Camera permission is required to scan fruits and vegetables.
        </Text>
        <Button title="Allow Camera" onPress={requestPermission} />
      </View>
    );
  }

  if (cameraVisible) {
    if (photoUri) {
      return (
        <View style={styles.cameraContainer}>
          <View
            style={styles.photoFrame}
            onLayout={({ nativeEvent }) => {
              const { width, height } = nativeEvent.layout;
              setPhotoFrameSize({ width, height });
            }}
          >
            <Image
              source={{ uri: photoUri }}
              style={styles.capturedImage}
              resizeMode="contain"
            />
            {detectionState === 'done' && boxScale > 0
              ? detections.map((detection, index) => {
                  const left =
                    photoOffsetX + detection.left * boxScale;
                  const top =
                    photoOffsetY + detection.top * boxScale;
                  const width =
                    (detection.right - detection.left) * boxScale;
                  const height =
                    (detection.bottom - detection.top) * boxScale;

                  return (
                    <View
                      key={`${detection.classId}-${index}`}
                      pointerEvents="none"
                      style={[
                        styles.detectionBox,
                        { left, top, width, height },
                      ]}
                    >
                      <Text
                        style={styles.detectionBoxLabel}
                        numberOfLines={1}
                        ellipsizeMode="tail"
                      >
                        {toDisplayName(detection.name)}
                      </Text>
                    </View>
                  );
                })
              : null}
          </View>

          <View style={styles.resultPanel}>
            <Text style={styles.resultTitle}>NUTRILENS</Text>
            <Text
              style={[
                styles.resultStatus,
                detectionState === 'error' && styles.errorText,
              ]}
            >
              {detectionState === 'processing'
                ? 'Detecting fruit or vegetable...'
                : detectionState === 'done' && detections.length > 0
                  ? 'Detection successful'
                  : detectionMessage}
            </Text>

            {detectionState === 'done' && detections.length > 0 ? (
              <ScrollView
                style={styles.detectionList}
                contentContainerStyle={styles.detectionListContent}
              >
                {detections.map((detection, index) => (
                  <Text
                    key={`${detection.classId}-${index}`}
                    style={styles.detectionText}
                  >
                    {index + 1}. Detected Food: {toDisplayName(detection.name)}
                    {'\n'}Confidence: {(detection.confidence * 100).toFixed(1)}%
                  </Text>
                ))}
              </ScrollView>
            ) : null}

            <View style={styles.actionRow}>
              <TouchableOpacity
                style={[
                  styles.scanAgainButton,
                  detectionState === 'processing' && styles.disabledButton,
                ]}
                disabled={detectionState === 'processing'}
                onPress={takeAnotherPhoto}
              >
                <Text style={styles.actionText}>TAKE AGAIN</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[
                  styles.closeButton,
                  detectionState === 'processing' && styles.disabledButton,
                ]}
                disabled={detectionState === 'processing'}
                onPress={closeScanner}
              >
                <Text style={styles.actionText}>CLOSE</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      );
    }

    return (
      <View style={styles.cameraContainer}>
        <CameraView
          style={styles.camera}
          facing="back"
          ref={cameraRef}
        />
        <View style={styles.cameraOverlay}>
          <Text style={styles.cameraTitle}>NUTRILENS</Text>
          <Text style={styles.cameraSubtitle}>Fruit &amp; Vegetable Scanner</Text>
          <View style={styles.scanBox}>
            <Text style={styles.scanHint}>Place a fruit or vegetable here</Text>
          </View>
          {cameraError ? (
            <Text style={[styles.cameraSubtitle, styles.errorText]}>
              {cameraError}
            </Text>
          ) : null}
          <Text style={styles.cameraStatus}>{modelStatus}</Text>
          <TouchableOpacity
            style={[
              styles.captureButton,
              modelState !== 'ready' && styles.disabledButton,
            ]}
            disabled={modelState !== 'ready'}
            onPress={capturePhoto}
            accessibilityLabel="Capture photo for detection"
          >
            <Text style={styles.captureText}>📷</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.cameraCloseButton} onPress={closeScanner}>
            <Text style={styles.actionText}>CLOSE</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>NUTRILENS</Text>
      <Text style={styles.subtitle}>Fruit &amp; Vegetable Scanner</Text>
      <Text style={styles.description}>
        Take a photo of a fruit or vegetable to identify it.
      </Text>
      <Text
        style={[
          styles.modelStatusText,
          modelState === 'error' && styles.errorText,
        ]}
      >
        {modelStatus}
      </Text>
      <TouchableOpacity
        style={[
          styles.scanButton,
          modelState === 'error' && styles.disabledButton,
        ]}
        disabled={modelState === 'error'}
        onPress={() => {
          clearPhoto();
          setCameraError('');
          setCameraVisible(true);
        }}
      >
        <Text style={styles.scanButtonText}>📷  SCAN FRUIT / VEGETABLE</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 25,
  },
  permissionContainer: {
    flex: 1,
    backgroundColor: '#ffffff',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 30,
  },
  cameraContainer: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  camera: {
    flex: 1,
  },
  photoFrame: {
    flex: 1,
    minHeight: 0,
    backgroundColor: '#0F172A',
    overflow: 'hidden',
  },
  capturedImage: {
    ...StyleSheet.absoluteFill,
    width: '100%',
    height: '100%',
  },
  cameraOverlay: {
    ...StyleSheet.absoluteFill,
    alignItems: 'center',
  },
  cameraTitle: {
    position: 'absolute',
    top: 64,
    alignSelf: 'center',
    fontSize: 26,
    fontWeight: 'bold',
    color: '#ffffff',
    backgroundColor: 'rgba(15, 23, 42, 0.65)',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 12,
    overflow: 'hidden',
  },
  detectionBox: {
    position: 'absolute',
    borderColor: '#2ecc71',
    borderWidth: 2,
  },
  detectionBoxLabel: {
    position: 'absolute',
    left: -2,
    top: -23,
    minWidth: 60,
    maxWidth: 150,
    color: '#ffffff',
    backgroundColor: '#2ecc71',
    fontSize: 12,
    fontWeight: 'bold',
    paddingHorizontal: 5,
    paddingVertical: 2,
    overflow: 'hidden',
  },
  cameraSubtitle: {
    position: 'absolute',
    top: 130,
    alignSelf: 'center',
    color: '#ffffff',
    fontSize: 16,
    textAlign: 'center',
  },
  scanBox: {
    position: 'absolute',
    top: '28%',
    width: '75%',
    height: '42%',
    borderWidth: 3,
    borderColor: '#2ecc71',
    borderRadius: 20,
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 15,
  },
  scanHint: {
    color: '#ffffff',
    backgroundColor: 'rgba(0,0,0,0.6)',
    padding: 8,
    borderRadius: 8,
  },
  captureButton: {
    position: 'absolute',
    bottom: 45,
    width: 75,
    height: 75,
    borderRadius: 40,
    backgroundColor: '#ffffff',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 5,
    borderColor: '#2ecc71',
  },
  captureText: {
    fontSize: 30,
  },
  cameraCloseButton: {
    position: 'absolute',
    top: 44,
    left: 18,
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    paddingVertical: 9,
    paddingHorizontal: 13,
    borderRadius: 10,
  },
  cameraStatus: {
    position: 'absolute',
    bottom: 130,
    color: '#ffffff',
    fontSize: 14,
    backgroundColor: 'rgba(15, 23, 42, 0.65)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
  },
  resultPanel: {
    paddingHorizontal: 18,
    paddingTop: 8,
    paddingBottom: 12,
    backgroundColor: '#ffffff',
  },
  resultTitle: {
    color: '#2ecc71',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 4,
  },
  resultStatus: {
    color: '#0F172A',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 8,
  },
  errorText: {
    color: '#ff6b6b',
  },
  detectionList: {
    maxHeight: 80,
    marginBottom: 10,
  },
  detectionListContent: {
    alignItems: 'center',
  },
  detectionText: {
    color: '#0F172A',
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
    marginBottom: 8,
  },
  actionRow: {
    flexDirection: 'row',
    gap: 12,
  },
  scanAgainButton: {
    flex: 1,
    backgroundColor: '#2ecc71',
    alignItems: 'center',
    paddingVertical: 14,
    borderRadius: 24,
  },
  closeButton: {
    flex: 1,
    backgroundColor: '#ff6b6b',
    alignItems: 'center',
    paddingVertical: 14,
    borderRadius: 24,
  },
  actionText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#2ecc71',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 18,
    color: '#555555',
    marginBottom: 20,
  },
  description: {
    fontSize: 16,
    color: '#666666',
    textAlign: 'center',
    marginBottom: 25,
  },
  modelStatusText: {
    fontSize: 15,
    color: '#2ecc71',
    marginBottom: 20,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  scanButton: {
    backgroundColor: '#2ecc71',
    paddingVertical: 17,
    paddingHorizontal: 30,
    borderRadius: 30,
  },
  scanButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  disabledButton: {
    opacity: 0.5,
  },
});
