import createFetchClient from 'openapi-fetch'
import createClient from 'openapi-react-query'

import type { paths } from './schema.d.ts'

export const fetchClient = createFetchClient<paths>({
  baseUrl: '',
})

export const api = createClient(fetchClient)
