import { ErrorBoundary } from 'react-error-boundary'
import './App.css'
import FaceRecognizer from './FaceRecognizer'
import ErrorFallback from './ErrorFallback'

function App() {
  return (
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onError={(error) => {
        console.error(error)
      }}
    >
      <FaceRecognizer />
    </ErrorBoundary>
  )
}

export default App
