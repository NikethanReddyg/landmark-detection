import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import ResultViewer from "./components/ResultViewer";
import "bootstrap/dist/css/bootstrap.min.css";

export default function App() {
  const [results, setResults] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);

  const handleUpload = (file, resultData) => {
    setResults(resultData);
    setImageUrl(URL.createObjectURL(file));
  };

  return (
    <div className="container py-5">
      <h2 className="mb-4">🗺️ Landmark Detection</h2>
      <UploadForm onResults={handleUpload} />

      {results && imageUrl && (
        <ResultViewer
          imageUrl={imageUrl}
          custom={results.custom}
          faster_rcnn={results.faster_rcnn}
        />
      )}

    </div>
  );
}
