// celebrityanalysis.ts
import { loadTensorflowModel } from 'react-native-fast-tflite';
import { supabase } from './supabase'; // Your already-initialized Supabase client
import { uploadImageToSupabase } from '@/helpers/ImageFunctions';
import { Platform } from 'react-native';
import RNFS from 'react-native-fs';
import ImageResizer from 'react-native-image-resizer';
import { GLView } from 'expo-gl';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';
import { UploadCelebrityAnalysis } from './UploadCelebrityAnalysis';

/** Minimal Photo interface */
export interface Photo {
  id: string;
  uri: string;
  label: string;
}

/** Constants for default top results and maximum progress. */
export const DEFAULT_TOP_N = 8;
// Steps are 0-indexed: 0, 1, 2, 3.
const MAX_PROGRESS = 3;

/** Bucket name for user photo uploads. */
const SUPABASE_PHOTO_BUCKET = "analysis-photos";

/** Define the model path using require() so it's bundled. */
const MODEL_PATH = require("../assets/models/facenet.tflite");

/**
 * Loads the MobileFaceNet model.
 */
async function loadModel(): Promise<any> {
  try {
    console.log("[Stage 1] Loading model...");
    const model = await loadTensorflowModel(MODEL_PATH);
    console.log("[Stage 1] Model loaded successfully.");
    return model;
  } catch (err) {
    console.error("[Stage 1] Error loading model:", err);
    throw err;
  }
}

/**
 * Resizes the image to 160x160 pixels.
 *
 * @param imageUri - The original image URI.
 * @returns A Promise that resolves to the URI of the resized image.
 */
async function resizeImageToFaceSize(imageUri: string): Promise<string> {
  try {
    console.log("[Stage 0] Resizing image to 160x160...");
    const resized = await ImageResizer.createResizedImage(
      imageUri,
      160,
      160,
      'JPEG',
      100
    );
    console.log(`[Stage 0] Image resized to 160x160: ${resized.uri}`);
    return resized.uri;
  } catch (error) {
    console.error("[Stage 0] Error resizing image:", error);
    throw error;
  }
}

/**
 * Uses Expo GL to extract raw RGB pixel data from a 160x160 image.
 *
 * @param imageUri - The URI of a 160x160 image.
 * @param width - Expected width (160)
 * @param height - Expected height (160)
 * @returns A Promise that resolves to a Uint8Array containing RGB pixel data.
 */
async function getImagePixelDataUsingExpoGL(
  imageUri: string,
  width: number,
  height: number
): Promise<Uint8Array> {
  console.log("[ExpoGL] Downloading asset from", imageUri);
  const asset = Asset.fromURI(imageUri);
  await asset.downloadAsync();
  console.log("[ExpoGL] Asset downloaded:", asset.localUri);

  console.log("[ExpoGL] Creating offscreen GL context...");
  const gl = await GLView.createContextAsync();
  console.log("[ExpoGL] GL context created.");

  const pixelBuffer = new Uint8Array(width * height * 4);
  console.log("[ExpoGL] Reading pixels...");
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixelBuffer);
  gl.endFrameEXP();
  console.log("[ExpoGL] Pixels read, buffer length:", pixelBuffer.length);

  // Convert RGBA to RGB.
  const rgbData = new Uint8Array(width * height * 3);
  let rgbIndex = 0;
  for (let i = 0; i < pixelBuffer.length; i += 4) {
    rgbData[rgbIndex++] = pixelBuffer[i];     // R
    rgbData[rgbIndex++] = pixelBuffer[i + 1];   // G
    rgbData[rgbIndex++] = pixelBuffer[i + 2];   // B
  }
  console.log("[ExpoGL] Extracted raw RGB data, length:", rgbData.length);
  return rgbData;
}

/**
 * Converts a Uint8Array of pixel data to a normalized Float32Array.
 * Normalization: (pixel - 127.5) / 127.5 maps [0,255] to [-1, 1].
 *
 * @param pixelData - The raw RGB pixel data.
 * @returns A normalized Float32Array.
 */
function normalizePixelData(pixelData: Uint8Array): Float32Array {
  console.log("[Stage 0] Normalizing pixel data...");
  const normalized = new Float32Array(pixelData.length);
  for (let i = 0; i < pixelData.length; i++) {
    normalized[i] = (pixelData[i] - 127.5) / 127.5;
  }
  console.log("[Stage 0] Normalization complete.");
  return normalized;
}

/**
 * Computes the face embedding for a given image.
 * Workflow:
 *   0. Resize image, extract pixel data via Expo GL, and normalize.
 *   1. Load the model.
 *   2. Run inference.
 *   3. Inference complete.
 *
 * @param imageUri - The user's face image URI.
 * @param onProgress - Optional progress callback (steps 0 to 3).
 * @returns A Promise that resolves to a 512-dimensional embedding array.
 */
export async function computeFaceEmbedding(
  imageUri: string,
  onProgress?: (step: number) => void
): Promise<number[]> {
  try {
    console.log(`Starting face embedding computation for image ${imageUri}`);
    if (onProgress) onProgress(0);

    // Verify that the file exists.
    const fileExists = await RNFS.exists(imageUri);
    console.log(`File exists at ${imageUri}: ${fileExists}`);
    if (!fileExists) {
      throw new Error(`Image file not found at ${imageUri}`);
    }

    // Resize the image.
    const resizedImageUri = await resizeImageToFaceSize(imageUri);
    console.log("[Stage 0] Resized image URI:", resizedImageUri);

    // Extract pixel data via Expo GL using new dimensions.
    const rawPixelData = await getImagePixelDataUsingExpoGL(resizedImageUri, 160, 160);
    console.log("[Stage 0] Raw pixel data obtained, length:", rawPixelData.length);

    // Normalize the pixel data.
    const normalizedTensor = normalizePixelData(rawPixelData);
    console.log("[Stage 0] Normalized tensor length:", normalizedTensor.length);

    // Load the model.
    const model = await loadModel();
    if (onProgress) onProgress(1);

    console.log("[Stage 2] Running model inference with normalized tensor...");
    const output = await model.run([normalizedTensor]);
    if (onProgress) onProgress(3);

    if (!output || !Array.isArray(output)) {
      console.error("Unexpected model output format:", output);
      throw new Error("Model output is not in expected format");
    }

    console.log(`[Stage 3] Raw embedding generated, length: ${output[0].length}`);
 
    if (output[0].length !== 128) {
      throw new Error(`Computed embedding length ${output[0].length} does not match expected 512`);
    }

    return output[0];
  } catch (error) {
    console.error(`Error computing embedding for image ${imageUri}:`, error);
    throw error;
  }
}

/**
 * Computes Euclidean distance between two numeric vectors.
 */
function euclideanDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error("Vectors must have the same length");
  return Math.sqrt(a.reduce((sum, value, i) => sum + Math.pow(value - b[i], 2), 0));
}

/**
 * Retrieves the total record count.
 */
async function getTotalRecordCount(): Promise<number> {
  const { data, error } = await supabase
    .from('celebritydataset_count')
    .select('record_count')
    .single();
  if (error) {
    console.error("Error fetching record count:", error);
    throw new Error("Failed to get record count");
  }
  return data.record_count;
}

/**
 * Fetches a chunk of celebrity records using pagination.
 */
async function getCelebrityChunk(offset: number, limit: number = 1000): Promise<any[]> {
  const { data, error } = await supabase
    .from('celebritydataset')
    .select('name, embedding, image_url')
    .range(offset, offset + limit - 1);
  if (error) {
    console.error("Error fetching chunk:", error);
    throw new Error("Failed to fetch data chunk");
  }
  return data;
}

/**
 * Searches the celebrity dataset by comparing embeddings.
 *
 * For normalization, all candidate distances are collected, then the normalized
 * similarity scores are computed so that the candidate with the smallest distance
 * gets a score of 10 and the one with the largest distance gets a score of 0.
 *
 * @param userEmbedding - A 512-dimensional embedding vector.
 * @param numResults - Number of top results to return.
 * @returns A JSON object with a "results" array.
 */
async function searchWithEmbedding(
  userEmbedding: number[],
  numResults: number = DEFAULT_TOP_N
): Promise<{ results: { rank: number; name: string; similarity: number; image_url: string }[] }> {
  const totalRecords = await getTotalRecordCount();
  console.log("Total records:", totalRecords);

  const chunkSize = 1000;
  let offset = 0;
  // Array to collect results with computed distances.
  const candidates: { name: string; image_url: string; distance: number }[] = [];

  while (offset < totalRecords) {
    console.log(`Fetching records ${offset} to ${offset + chunkSize - 1}...`);
    const chunk = await getCelebrityChunk(offset, chunkSize);
    for (const record of chunk) {
      const name = record.name;
      // Skip duplicate names.
      if (candidates.find(c => c.name === name)) continue;
      const embedding: number[] = record.embedding;
      if (embedding.length !== userEmbedding.length) {
        console.error(`Length mismatch for celebrity "${name}": expected ${userEmbedding.length}, got ${embedding.length}`);
        throw new Error("Vectors must have the same length");
      }
      const distance = euclideanDistance(userEmbedding, embedding);
      console.log(`[Search] Celebrity ${name}: Euclidean distance = ${distance.toFixed(4)}`);
      candidates.push({ name, image_url: record.image_url, distance });
    }
    offset += chunkSize;
  }

  // Compute minimum and maximum distances among candidates.
  const distances = candidates.map(c => c.distance);
  const minDistance = Math.min(...distances);
  const maxDistance = Math.max(...distances);
  console.log(`Min Distance: ${minDistance.toFixed(4)}, Max Distance: ${maxDistance.toFixed(4)}`);

  // Function for min–max normalization.
  function computeNormalizedScore(distance: number): number {
    if (maxDistance === minDistance) return 10;
    return 10 * ((maxDistance - distance) / (maxDistance - minDistance));
  }

  // Compute normalized similarity scores for each candidate.
  const candidatesWithScore = candidates.map(c => {
    const score = computeNormalizedScore(c.distance);
    console.log(`[Search] Celebrity ${c.name}: Normalized similarity score = ${score}`);
    return { ...c, score };
  });

  // Sort candidates by descending normalized similarity and select the top results.
  const sortedResults = candidatesWithScore.sort((a, b) => b.score - a.score);
  const topResults = sortedResults.slice(0, numResults).map((result, index) => ({
    rank: index + 1,
    name: result.name,
    similarity: parseFloat(result.score.toFixed(2)),
    image_url: result.image_url,
  }));

  return { results: topResults };
}

/**
 * Processes the user's photo: computes the embedding, uploads the user's main photo,
 * and then searches the celebrity dataset.
 *
 * Since the function is expected to receive only one photo, we process only the first photo.
 *
 * @param photos - An array of Photo objects (only the first photo is processed).
 * @param userId - The user's ID.
 * @param onProgress - Callback for updating progress (steps 0 to 3).
 * @param numResults - Number of top similarity results to return.
 * @returns A JSON object with "userData" and "results".
 */
export async function analyzeUserPhotos(
  photos: Photo[],
  userId: string,
  onProgress: (step: number) => void,
  numResults: number = DEFAULT_TOP_N
): Promise<{ userData: { userId: string; photo_url: string }, results: { rank: number; name: string; similarity: number; image_url: string }[] }> {
  if (photos.length === 0) throw new Error("No photos provided.");

  // Process only the first photo.
  const photo = photos[0];
  console.log(`Processing photo: ${photo.label}`);
  const embedding = await computeFaceEmbedding(photo.uri, (step: number) => {
    console.log(`Progress: Step ${step}`);
    onProgress(step);
  });
  console.log(`Photo embedding length: ${embedding.length}`);

  // Upload the user's photo and get the URL.
  const userPhotoUrl = (await uploadImageToSupabase([photo], SUPABASE_PHOTO_BUCKET))[0];
  console.log("User photo URL obtained:", userPhotoUrl);

  // Search the celebrity dataset using the computed embedding.
  const searchResult = await searchWithEmbedding(embedding, numResults);
  
  // Assemble the complete analysis object.
  const analysis = {
    userData: { userId, photo_url: userPhotoUrl },
    results: searchResult.results,
  };

  // Upload the analysis to Supabase.
  await UploadCelebrityAnalysis(analysis, userId);

  return analysis;
}