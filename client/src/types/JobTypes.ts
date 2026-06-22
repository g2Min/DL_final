export interface JobResult {
  combo_idx: number
  image_url: string
}

export interface JobStatus {
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: string[]
  results: JobResult[] | null
  error: string | null
}