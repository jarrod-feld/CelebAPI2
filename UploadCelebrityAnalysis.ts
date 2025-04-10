// uploadCelebrityAnalysis.ts
import { supabase } from '@/lib/supabase'; // Supabase now lives in @/lib
import { getCelebrityAnalysisCache, setCelebrityAnalysisCache } from '@/services/celebritycache';

/**
 * Uploads the user's celebrity analysis to Supabase.
 *
 * This function stores:
 * - analysis: the JSON result from the celebrity matching process
 * - user_id: the id of the user who performed the analysis
 *
 * After a successful upload, the function updates the local celebrity analysis cache.
 *
 * @param analysis - The analysis JSON data.
 * @param userId - The user's unique identifier.
 * @returns The inserted record data from Supabase.
 */
export async function UploadCelebrityAnalysis(analysis: any, userId: string): Promise<any> {
  console.log("Uploading celebrity analysis for user:", userId);
  const { data, error } = await supabase
    .from('celebrity_analysis')
    .insert([
      {
        user_id: userId,
        analysis: analysis,
      },
    ]);

  if (error) {
    console.error('Error uploading celebrity analysis:', error);
    throw error;
  }

  console.log('Celebrity analysis uploaded successfully:', data);

  // Update the local celebrity analysis cache.
  try {
    const currentCache = await getCelebrityAnalysisCache();
    console.log("Current celebrity cache before update:", currentCache);
    // Assuming data returns an array with one record.
    const newRecord = data[0];
    const updatedCache = [newRecord, ...currentCache];
    await setCelebrityAnalysisCache(updatedCache);
    console.log('Local celebrity analysis cache updated with new record:', newRecord);
  } catch (cacheError) {
    console.error('Error updating local celebrity analysis cache:', cacheError);
  }

  return data;
}
