// eslint-disable-next-line @typescript-eslint/no-explicit-any
export default function ErrorFallback({ error }: { error: any }) {
  return (
    <div>
      <p>Something went wrong 😭</p>
      {error.message && <span>Here's the error: {error.message}</span>}
    </div>
  )
}
