import {
  FaceMatcher,
  nets,
  TNetInput,
  fetchImage,
  detectSingleFace,
  LabeledFaceDescriptors,
  detectAllFaces,
  resizeResults,
  draw
} from 'face-api.js'
import { useRef, useState, useEffect, useCallback } from 'react'
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
            const referenceImage: TNetInput = await fetchImage(imgConfig.uri)
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

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot() ?? ''
    setImg(imageSrc)
  }, [])

  useInterval(() => {
    capture()
  }, 3000)

  /**
   * process the image we captured from the webcam
   * @returns
   */
  async function handleImageLoad() {
    const image = imgRef.current
    const canvas = canvasRef.current
    if (image && canvas) {
      const detections = await detectAllFaces(image)
        .withFaceLandmarks()
        .withFaceDescriptors()
      const resizedDimensions = resizeResults(detections, {
        height: canvas.height,
        width: canvas.width
      })
      const context = canvas.getContext('2d')
      context?.clearRect(0, 0, canvas.width, canvas.height)
      context?.drawImage(image, 0, 0, canvas.width, canvas.height)

      if (!detections.length) {
        setDetectedPerson((prev) => {
          return {
            ...prev,
            score: 1,
            person: 'Person not found'
          }
        })
        return
      }
      draw.drawDetections(canvas, resizedDimensions)
      // person recognition from all faces in current screenshot
      detections.forEach((fd) => {
        const bestMatch = faceDetectorInstance.current?.findBestMatch(
          fd.descriptor
        )
        setDetectedPerson((prev) => {
          return {
            ...prev,
            score: bestMatch?.distance ?? 0,
            person: bestMatch?.label ?? ''
          }
        })
      })
    }
  }

  if (loading) {
    return <p>Loading...</p>
  }
  return (
    <>
      {' '}
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
      <h3>Score: {detectedPerson.score}</h3>
      <canvas ref={canvasRef} height={200} width={200}></canvas>
    </>
  )
}
