export interface HealthResponse {
  status: string
  version: string
  environment: string
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/health`)

  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`)
  }

  return (await response.json()) as HealthResponse
}
