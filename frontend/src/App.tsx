import { useState, useEffect } from 'react'

type Detection = {
  id: number
  camera_id: string
  object_key: string
  created_at: string
}

const App = () => {
  const [detections, setDetections] = useState<Detection[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadDetections = async () => {
      try {
        const response = await fetch('/detections?limit=50')
        if (!response.ok) {
          setError(`Failed to load detections ${response.statusText}`)
          return
        }
        const rows: Detection[] = await response.json()
        setDetections(rows)
      } catch (error) {
        setError(
          `Failed to load detections: ${error instanceof Error ? error.message : 'Unknown error'}`,
        )
      }
    }
    void loadDetections()
  }, [])

  if (error) {
    return <main>Error: {error}</main>
  }

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

export default App
