import {
  FaceMatcher,
  nets,
  fetchImage,
  detectSingleFace,
  LabeledFaceDescriptors,
  resizeResults,
  draw
} from 'face-api.js'
import { useRef, useState, useEffect } from 'react'
import Webcam from 'react-webcam'
import { useInterval } from './hooks'

const referenceImages: { label: string; uri: string }[] = [
  { label: 'Joseph Hailu', uri: './me.jpg' },
  { label: 'Nicolas Cage', uri: './Nicolas_Cage.jpeg' },
  { label: 'Nicolas Cage', uri: './Nicolas_Cage.webp' }
]

const videoConstraints = {
  width: 200,
  height: 200,
  facingMode: 'user'
}

export default function FaceRecognizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const webcamRef = useRef<Webcam>(null)
  const faceDetectorInstance = useRef<FaceMatcher | null>(null)
  const context = canvasRef.current?.getContext('2d', {
    willReadFrequently: true
  })
  /** use this to periodically collect 'stills' from the webcam and feed into facedector */
  const [img, setImg] = useState('')
  const [loading, setLoading] = useState(false)
  const [detectedPerson, setDetectedPerson] = useState({
    person: '',
    score: 0
  })

  useEffect(() => {
    async function loadModelsAndBuildFaceMatcher() {
      try {
        setLoading(true)
        // very big files, please take care to do only once
        await Promise.all([
          nets.ssdMobilenetv1.loadFromUri('./weights'),
          nets.faceLandmark68Net.loadFromUri('./weights'),
          nets.faceRecognitionNet.loadFromUri('./weights')
        ])

        // build a list of label and [descriptor] to feed into
        // the FaceMatcher instance
        const labeledDescriptors = await Promise.all(
          referenceImages.map(async (imgConfig) => {
            const referenceImage = await fetchImage(imgConfig.uri)
            const result = await detectSingleFace(referenceImage)
              .withFaceLandmarks()
              .withFaceDescriptor()
            if (!result) {
              return
            }
            return new LabeledFaceDescriptors(imgConfig.label, [
              result.descriptor
            ])
          })
        )
        faceDetectorInstance.current = new FaceMatcher(labeledDescriptors)
        setLoading(false)
      } catch (error) {
        setLoading(false)
        console.error(error)
      }
    }

    loadModelsAndBuildFaceMatcher()
  }, [])
  useInterval(() => {
    const imageSrc = webcamRef.current?.getScreenshot() ?? ''
    setImg(imageSrc)
  }, 3000)

  /**
   * process the image we captured from the webcam
   * @returns
   */
  async function handleImageLoad() {
    const image = imgRef.current
    const canvas = canvasRef.current
    if (image && canvas && context) {
      const detection = await detectSingleFace(image)
        .withFaceLandmarks()
        .withFaceDescriptor()

      context.clearRect(0, 0, canvas.width, canvas.height)
      context.drawImage(image, 0, 0, canvas.width, canvas.height)

      if (!detection) {
        setDetectedPerson((prev) => {
          return {
            ...prev,
            score: 1,
            person: 'Person not found'
          }
        })
        return
      }
      const resizedDimensions = resizeResults(detection, {
        height: canvas.height,
        width: canvas.width
      })
      draw.drawDetections(canvas, [resizedDimensions?.detection])
      // person recognition from all faces in current screenshot
      const bestMatch = faceDetectorInstance.current?.findBestMatch(
        detection.descriptor
      )
      setDetectedPerson((prev) => {
        return {
          ...prev,
          score: bestMatch?.distance ?? 0,
          person: bestMatch?.label ?? ''
        }
      })
    }
  }

  if (loading) {
    return <p>Loading...</p>
  }
  return (
    <>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat='image/jpeg'
        videoConstraints={videoConstraints}
        style={{ opacity: 0.0, position: 'absolute', zIndex: -1000000 }}
      />
      <img
        ref={imgRef}
        src={img}
        height={200}
        width={200}
        style={{ opacity: 0.0, position: 'absolute', zIndex: -1 }}
        onLoad={handleImageLoad}
      />
      <h3>Person: {detectedPerson.person}</h3>
      <h3>Euclidean Distance (lower is better): {detectedPerson.score}</h3>
      <canvas ref={canvasRef} height={200} width={200}></canvas>
    </>
  )
}
