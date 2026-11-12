import { useDetectionStream } from '../api/use-detection-stream.ts'
import { useDetections } from '../api/get-detections.ts'

export const DetectionsPage = () => {
  const detectionsQuery = useDetections()
  useDetectionStream(detectionsQuery.isSuccess)

  if (detectionsQuery.isError) {
    const message =
      detectionsQuery.error instanceof Error ? detectionsQuery.error.message : 'Unknown error'
    return <main>Error: {message}</main>
  }

  const detections = detectionsQuery.data ?? []

  return (
    <main>
      <h1>Vision events</h1>
      <ul>
        {detections.map((detection) => (
          <li key={detection.id}>
            <div>{detection.camera_id}</div>
            <div>{detection.object_key}</div>
            <div>{new Date(detection.created_at).toLocaleString()}</div>
          </li>
        ))}
      </ul>
    </main>
  )
}
