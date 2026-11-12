import { useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'

import { type Detection, detectionsQueryOptions } from './get-detections.ts'

const prependDetection = (current: Detection[], row: Detection): Detection[] => {
  if (current.some((item) => item.id === row.id)) {
    return current
  }
  return [row, ...current]
}

export const useDetectionStream = (enabled: boolean) => {
  const queryClient = useQueryClient()

  useEffect(() => {
    if (!enabled) {
      return
    }

    const source = new EventSource('/detections/stream')
    const handleMessage = (event: MessageEvent<string>) => {
      const row = JSON.parse(event.data) as Detection
      queryClient.setQueryData<Detection[]>(detectionsQueryOptions.queryKey, (current) =>
        prependDetection(current ?? [], row),
      )
    }
    source.onmessage = handleMessage

    return () => {
      source.close()
    }
  }, [enabled, queryClient])
}
