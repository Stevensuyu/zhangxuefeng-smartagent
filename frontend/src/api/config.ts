import { API_BASE } from '../config'

export interface AIConfig {
  api_key: string
  base_url: string
  model: string
  has_api_key: boolean
}

const STORAGE_KEY = 'zhangxuefeng-agent:ai-config'

export function getStoredConfig(): AIConfig {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (raw) {
      return JSON.parse(raw) as AIConfig
    }
  } catch {
    // ignore
  }
  return {
    api_key: '',
    base_url: 'https://api.openai.com/v1',
    model: 'gpt-4o-mini',
    has_api_key: false,
  }
}

export function saveConfig(config: Partial<AIConfig>): void {
  const current = getStoredConfig()
  const updated = { ...current, ...config }
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
}

export async function fetchConfig(): Promise<AIConfig> {
  try {
    const response = await fetch(`${API_BASE}/api/config`)
    if (response.ok) {
      const data = await response.json() as AIConfig
      saveConfig(data)
      return data
    }
  } catch {
    // ignore
  }
  return getStoredConfig()
}

export async function updateConfig(config: {
  api_key?: string
  base_url?: string
  model?: string
}): Promise<AIConfig> {
  try {
    const response = await fetch(`${API_BASE}/api/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
    if (response.ok) {
      const data = await response.json()
      saveConfig(data.config)
      return data.config
    }
  } catch {
    // ignore
  }
  return getStoredConfig()
}
