const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Allow .tflite model files to be bundled with the app
config.resolver.assetExts.push('tflite');

module.exports = config;