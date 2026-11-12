import { useQuery } from '@tanstack/react-query'

import { api, fetchClient } from '@/api/client.ts'
import type { components } from '@/api/schema.d.ts'

export type Detection = components['schemas']['DetectionResponseDto']

const detectionsRequest = {
  params: {
    query: {
      limit: 50,
    },
  },
}

export const detectionsQueryOptions = api.queryOptions('get', '/detections', detectionsRequest)

export const useDetections = () =>
  useQuery({
    queryKey: detectionsQueryOptions.queryKey,
    queryFn: async ({ signal }) => {
      const { data, response } = await fetchClient.GET('/detections', {
        ...detectionsRequest,
        signal,
      })
      if (!response.ok || data === undefined) {
        throw new Error(`Failed to load detections ${response.status}`)
      }
      return data
    },
  })
