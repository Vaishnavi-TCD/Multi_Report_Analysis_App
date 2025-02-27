import { useState } from "react";
import axios from "axios";

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState(false); 
  const [loading, setLoading] = useState(false); // ✅ Loading state for queries

  // ✅ Handle file selection
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  // ✅ Upload file and trigger FAISS indexing
  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      console.log("📤 Uploading file:", selectedFile.name);
      const uploadResponse = await axios.post("http://127.0.0.1:8000/upload/", formData);

      console.log("✅ Upload Response:", uploadResponse.data);

      if (uploadResponse.data.message === "File uploaded successfully") {
        alert("File uploaded and indexed successfully!");
        setUploadSuccess(true);
      } else {
        alert("File upload failed. Try again.");
      }
    } catch (error) {
      console.error("❌ Upload Error:", error);
      alert("Error uploading file.");
    }
  };

  const handleQuery = async () => {
    if (!uploadSuccess) {
      alert("Please upload a file first before querying!");
      return;
    }
  
    setLoading(true); // ✅ Show loading state
    setResponse("");  // ✅ Reset response
  
    try {
      console.log("🔍 Sending Query:", {
        report_name: selectedFile.name,
        query: query.trim(),
      });
  
      const queryResponse = await axios.post("http://127.0.0.1:8000/query/", {
        report_name: selectedFile.name,
        query: query.trim(),
      });
  
      console.log("📊 Retrieved FAISS Data:", queryResponse.data);
  
      // ✅ Debug: Log full API response
      console.log("✅ Full API Response:", queryResponse);
  
      if (!queryResponse.data || !queryResponse.data.response) {
        console.error("❌ Error: API returned invalid response format");
        setResponse("No relevant results found.");
        setLoading(false);
        return;
      }
  
      // ✅ Extract and clean the response
      const retrievedResponse = queryResponse.data.response.trim();
  
      console.log("✅ Extracted Response:", retrievedResponse);
  
      setTimeout(() => {
        setResponse(retrievedResponse);
        setLoading(false);
      }, 500); // Small delay for UI update
    } catch (error) {
      console.error("❌ Query Error:", error);
      alert("Error fetching query response.");
      setLoading(false);
    }
  };
  
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
      <h1 className="text-4xl font-bold text-blue-600 mb-6">AI-Powered Report Analysis</h1>

      {/* ✅ File Upload Section */}
      <div className="mb-4">
        <input type="file" onChange={handleFileChange} className="border p-2 rounded" />
        <button onClick={handleUpload} className="ml-2 px-4 py-2 bg-gray-200 border rounded hover:bg-gray-300">
          Upload Report
        </button>
      </div>

      {/* ✅ Query Input Section */}
      <div className="mb-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your question..."
          className="border p-2 rounded w-80"
        />
        <button
          onClick={handleQuery}
          className="ml-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700"
        >
          Get Insights
        </button>
      </div>

      {/* ✅ AI Response Section */}

      <div className="mt-4 w-3/4 p-4 bg-white shadow-md rounded">
        <h2 className="text-xl font-bold">AI Response:</h2>

        {/* ✅ Show loading spinner */}
        {loading ? (
          <p className="text-gray-500 mt-2">🔄 Processing your query...</p>
        ) : (
          <p className="text-gray-700 mt-2">{response || "No response yet..."}</p>
        )}
      </div>

    </div>
  );
}
