// faceRecognition.ts

import { createClient } from "@supabase/supabase-js";
import { ParquetReader } from "parquetjs-lite";
import * as faceapi from "face-api.js"; // Ensure models are loaded as required

export interface CelebrityMatch {
  rank: number;
  name: string;
  similarity: number;
  distance: number;
  image: string; // base64-encoded image string
}

export interface CelebrityData {
  names: string[];
  embeddings: number[][];
  images: string[];
}

// Supabase configuration using Expo environment variable names.
const SUPABASE_URL = process.env.EXPO_PUBLIC_SUPABASE_URL || "YOUR_SUPABASE_URL";
const SUPABASE_KEY = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY || "YOUR_SUPABASE_KEY";
const BUCKET = "AnalysisLib";
const FILE_PATH = "combined_celebrity_embeddings.parquet";

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

/**
 * Retrieves the public URL for the Parquet file from Supabase.
 */
async function getPublicParquetUrl(): Promise<string> {
  const { data, error } = supabase.storage.from(BUCKET).getPublicUrl(FILE_PATH);
  if (error) {
    throw error;
  }
  return data.publicUrl;
}

/**
 * Streams the Parquet file from Supabase, converts it into a Buffer,
 * and then loads the celebrity data.
 */
export async function loadCelebrityData(): Promise<CelebrityData> {
  const publicUrl = await getPublicParquetUrl();
  const response = await fetch(publicUrl);
  if (!response.body) {
    throw new Error("ReadableStream not supported in this environment");
  }
  // Stream the response into a single buffer.
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let receivedLength = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      receivedLength += value.length;
    }
  }
  const buffer = new Uint8Array(receivedLength);
  let position = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, position);
    position += chunk.length;
  }
  const nodeBuffer = Buffer.from(buffer);
  // Open the Parquet reader from the in-memory buffer.
  const parquetReader = await ParquetReader.openBuffer(nodeBuffer);
  const cursor = parquetReader.getCursor();
  const names: string[] = [];
  const embeddings: number[][] = [];
  const images: string[] = [];
  let record: any;
  while ((record = await cursor.next())) {
    names.push(record.name);
    embeddings.push(record.embedding);
    images.push(record.image);
  }
  await parquetReader.close();
  return { names, embeddings, images };
}

/**
 * Computes the Euclidean distance between two vectors.
 */
export function euclideanDistance(vec1: number[], vec2: number[]): number {
  if (vec1.length !== vec2.length) {
    throw new Error("Vectors must be of the same length");
  }
  let sum = 0;
  for (let i = 0; i < vec1.length; i++) {
    const diff = vec1[i] - vec2[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Given a test face embedding and celebrity data, returns the top matching celebrities.
 */
export function getTopMatches(
  testEmbedding: number[],
  celebrityData: CelebrityData,
  topN: number = 3
): CelebrityMatch[] {
  const { names, embeddings, images } = celebrityData;
  const distances = embeddings.map((embedding) =>
    euclideanDistance(embedding, testEmbedding)
  );
  const sortedIndices = distances
    .map((distance, index) => ({ distance, index }))
    .sort((a, b) => a.distance - b.distance)
    .map((item) => item.index);
  const matches: CelebrityMatch[] = sortedIndices
    .slice(0, topN)
    .map((idx, rank) => {
      const distance = distances[idx];
      const similarity = 1 / (1 + distance);
      return {
        rank: rank + 1,
        name: names[idx],
        similarity: parseFloat(similarity.toFixed(4)),
        distance: parseFloat(distance.toFixed(4)),
        image: images[idx]
      };
    });
  return matches;
}

/**
 * Computes the face embedding for an image using face-api.js.
 * Ensure that you have loaded the required models beforehand.
 *
 * Example model loading:
 *   await faceapi.nets.ssdMobilenetv1.loadFromUri('/models');
 *   await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
 *   await faceapi.nets.faceRecognitionNet.loadFromUri('/models');
 *
 * @param image - An HTMLImageElement containing the face.
 * @returns A promise that resolves to the face embedding (an array of numbers).
 */
export async function computeFaceEmbedding(image: HTMLImageElement): Promise<number[]> {
  const detection = await faceapi
    .detectSingleFace(image)
    .withFaceLandmarks()
    .withFaceDescriptor();
  if (!detection) {
    throw new Error("No face detected in the image");
  }
  return detection.descriptor as number[];
}

/**
 * Given an HTMLImageElement, computes its face embedding,
 * loads celebrity embeddings from Supabase (streaming the Parquet file),
 * and returns the top 3 matching celebrities with their images.
 */
export async function findTopCelebrityMatches(
  image: HTMLImageElement
): Promise<CelebrityMatch[]> {
  // Compute the face embedding for the input image.
  const testEmbedding = await computeFaceEmbedding(image);
  // Load celebrity embeddings data from Supabase.
  const celebrityData = await loadCelebrityData();
  // Compute and return the top 3 matches.
  return getTopMatches(testEmbedding, celebrityData, 3);
}

/*
Example usage:

const imgElement = document.getElementById("inputImage") as HTMLImageElement;
findTopCelebrityMatches(imgElement)
  .then(matches => console.log("Top Matches:", matches))
  .catch(error => console.error("Error:", error));
*/
