import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import chatHandler from './api/chat.js'

function readRequestBody(req) {
  return new Promise((resolve, reject) => {
    let body = ''
    req.setEncoding('utf8')
    req.on('data', (chunk) => {
      body += chunk
    })
    req.on('end', () => {
      if (!body) {
        resolve({})
        return
      }

      try {
        resolve(JSON.parse(body))
      } catch (_) {
        reject(new Error('Request body must be valid JSON.'))
      }
    })
    req.on('error', reject)
  })
}

function localApiPlugin() {
  return {
    name: 'paperpulse-local-api',
    configureServer(server) {
      server.middlewares.use('/api/chat', async (req, res) => {
        try {
          req.body = await readRequestBody(req)
          res.status = (statusCode) => {
            res.statusCode = statusCode
            return res
          }

          await chatHandler(req, res)
        } catch (error) {
          if (res.headersSent) return
          const statusCode = error.message === 'Request body must be valid JSON.' ? 400 : 500
          res.statusCode = statusCode
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ error: error.message || 'Local API request failed.' }))
        }
      })
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  Object.entries(env).forEach(([key, value]) => {
    if (process.env[key] === undefined) process.env[key] = value
  })

  return {
    plugins: [react(), localApiPlugin()],
    server: {
      port: 5173,
      open: true,
    },
  }
})
