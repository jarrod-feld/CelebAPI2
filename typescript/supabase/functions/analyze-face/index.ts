import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { decode } from 'https://deno.land/x/base64/mod.ts'
import * as fr from 'npm:face-recognition'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req: Request) => {
  // Handle CORS
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? ''
    )

    // Only accept POST requests
    if (req.method !== 'POST') {
      return new Response(JSON.stringify({ error: 'Method not allowed' }), {
        status: 405,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      })
    }

    // Get form data
    const formData = await req.formData()
    const file = formData.get('file')
    const numResults = parseInt(formData.get('num_results')?.toString() || '8')

    if (!file || !(file instanceof File)) {
      throw new Error('No file provided')
    }

    // Process image
    const buffer = await file.arrayBuffer()
    const image = fr.loadImage(new Uint8Array(buffer))
    const encodings = fr.faceEncodings(image)

    if (encodings.length === 0) {
      throw new Error('No face detected in the image')
    }

    const userEncoding = encodings[0]

    // Get total records count
    const { data: countData, error: countError } = await supabaseClient
      .from('celebritydataset_count')
      .select('record_count')
      .single()

    if (countError) throw countError

    const totalRecords = countData.record_count
    const chunkSize = 1000
    let offset = 0
    const candidates = []

    // Process in chunks
    while (offset < totalRecords) {
      const { data: records, error } = await supabaseClient
        .from('celebritydataset')
        .select('name, embedding, image_url')
        .range(offset, offset + chunkSize - 1)

      if (error) throw error
      if (!records.length) break

      for (const record of records) {
        const distance = euclideanDistance(userEncoding, record.embedding)
        candidates.push({
          name: record.name,
          image_url: record.image_url,
          distance,
        })
      }

      offset += chunkSize
    }

    // Process results
    const results = processResults(candidates, numResults)

    return new Response(JSON.stringify({ results }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    })

  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    })
  }
})

function euclideanDistance(a: number[], b: number[]): number {
  return Math.sqrt(
    a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0)
  )
}

function processResults(candidates: any[], numResults: number) {
  const distances = candidates.map(c => c.distance)
  const minDistance = Math.min(...distances)
  const maxDistance = Math.max(...distances)

  candidates.forEach(c => {
    c.score = maxDistance === minDistance
      ? 10.0
      : 10 * (maxDistance - c.distance) / (maxDistance - minDistance)
  })

  return candidates
    .sort((a, b) => b.score - a.score)
    .slice(0, numResults)
    .map((c, i) => ({
      rank: i + 1,
      name: c.name,
      similarity: Math.round(c.score * 100) / 100,
      image_url: c.image_url
    }))
}