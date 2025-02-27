import React, { useState } from "react";
import axios from "axios";
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;


// Add this CSS at the beginning to reset browser default styles
const globalStyles = `
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body, html, #root {
    width: 100%;
    min-height: 100vh;
    overflow-x: hidden;
  }
`;

export default function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [comparisonResponse, setComparisonResponse] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [activeFile, setActiveFile] = useState(""); // Track the selected file for insights
  const [queryHistory, setQueryHistory] = useState([]); // Store query history

  // Allow multi-location file selection
  const handleFileChange = (event) => {
    const newFiles = [...selectedFiles, ...event.target.files];
    newFiles.forEach(file => {
      if (!file.name.endsWith(".txt") && !file.name.endsWith(".pdf")) {
          alert(`Unsupported file type: ${file.name}`);
          return;
      }
    });
    setSelectedFiles(newFiles);

    // Set first uploaded file as default selection
    if (!activeFile && newFiles.length > 0) {
      setActiveFile(newFiles[0].name);
    }
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      alert("Please select at least one file!");
      return;
    }

    let uploadCount = 0;
    for (const file of selectedFiles) {
      const formData = new FormData();
      formData.append("file", file);

      try {
        await axios.post("${BACKEND_URL}/upload/", formData);
        uploadCount++;
      } catch (error) {
        console.error("Upload Error:", error);
        alert(`Error uploading file: ${file.name}`);
      }
    }

    if (uploadCount > 0) {
      setUploadSuccess(true);
      alert("Files uploaded and indexed successfully!");
    }
  };

  const handleQuery = async () => {
    if (!uploadSuccess) {
      alert("Please upload a file first before querying!");
      return;
    }
    if (!query.trim()) {
      alert("Please enter a query before getting insights!");
      return;
    }
    if (!activeFile) {
      alert("Please select a report before querying insights!");
      return;
    }

    try {
      const queryResponse = await axios.post("${BACKEND_URL}/query/", {
        report_name: activeFile, // Query selected report
        query: query.trim(),
      }, {
        headers: { "Content-Type": "application/json" }
      });

      console.log("🔍 Query Response:", queryResponse.data);

      if (!queryResponse.data.response || queryResponse.data.response === "Report not found.") {
        setResponse("No relevant insights found for the given query.");
      } else {
        setResponse(queryResponse.data.response);
        setQueryHistory((prevHistory) => [...new Set([query.trim(), ...prevHistory])]); // Store unique queries
      }
    } catch (error) {
      console.error("Query Error:", error);
      alert("Error fetching query response.");
    }
  };

  const handleCompare = async () => {
    if (selectedFiles.length < 2) {
      alert("Please select at least two files to compare!");
      return;
    }
    if (!query.trim()) {
      alert("Please enter a query before comparing reports!");
      return;
    }

    try {
      const compareResponse = await axios.post("${BACKEND_URL}/compare/", {
        report1: selectedFiles[0].name,
        report2: selectedFiles[1].name,
        query: query.trim(),
      });

      setComparisonResponse(compareResponse.data.response);
      setQueryHistory((prevHistory) => [...new Set([query.trim(), ...prevHistory])]); // Store unique queries
    } catch (error) {
      console.error("Comparison Error:", error);
      alert("Error fetching comparison response.");
    }
  };

  const handleClearHistory = () => {
    setQueryHistory([]);
  };

  // Updated style to ensure full width
  const pageStyle = {
    background: "linear-gradient(to bottom, #6366f1, #3b82f6)",
    minHeight: "100vh",
    width: "100%",
    fontFamily: "Arial, sans-serif",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    textAlign: "center",
    padding: "20px",
    color: "#fff",
    margin: 0
  };

  const sectionStyle = {
    backgroundColor: "#fff",
    borderRadius: "12px",
    padding: "24px",
    marginBottom: "24px",
    width: "90%",
    maxWidth: "1200px", // Increased max-width for larger screens
    boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
    textAlign: "center",
  };

  const headerStyle = {
    width: "90%", // Match the section width
    maxWidth: "1200px", // Match the section max-width
    backgroundColor: "#4338ca",
    padding: "20px 0",
    marginBottom: "32px",
    borderRadius: "8px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
    textAlign: "center",
  };

  const buttonStyle = {
    backgroundColor: "#4f46e5",
    color: "white",
    border: "none",
    borderRadius: "8px",
    padding: "12px 24px",
    margin: "8px",
    cursor: "pointer",
    fontWeight: "bold",
    boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
    transition: "background-color 0.3s ease",
  };

  const inputStyle = {
    width: "100%",
    padding: "12px",
    margin: "8px 0",
    borderRadius: "8px",
    border: "1px solid #ccc",
  };

  const resultStyle = {
    backgroundColor: "#f3f4f6",
    padding: "16px",
    borderRadius: "8px",
    marginTop: "16px",
    textAlign: "left",
    color: "#1f2937",
  };

  return (
    <>
      {/* Add style tag for global styles */}
      <style>{globalStyles}</style>
      
      <div style={pageStyle}>
        <div style={headerStyle}>
          <h1 style={{ fontSize: "36px", fontWeight: "bold", margin: "0", color: "white" }}>
            Multi-Report Comparison
          </h1>
          <p style={{ color: "#c7d2fe", marginTop: "8px" }}>
            Upload, analyze, and compare your reports with ease
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={{ color: "#4338ca", fontSize: "24px", marginBottom: "16px" }}>Upload Reports</h2>
          
          <input 
            type="file" 
            multiple 
            onChange={handleFileChange} 
            style={{...inputStyle, backgroundColor: "#f3f4f6"}} 
          />
          
          <button 
            onClick={handleUpload} 
            style={{...buttonStyle, backgroundColor: "#2563eb", width: "100%"}}
          >
            Upload Reports
          </button>
        </div>

        {selectedFiles.length > 0 && (
          <div style={sectionStyle}>
            <h2 style={{ color: "#4338ca", fontSize: "24px", marginBottom: "16px" }}>Select Document</h2>
            
            <select 
              style={{...inputStyle, backgroundColor: "#f3f4f6"}}
              value={activeFile} 
              onChange={(e) => setActiveFile(e.target.value)}
            >
              {selectedFiles.map((file, index) => (
                <option key={index} value={file.name}>
                  {file.name}
                </option>
              ))}
            </select>
          </div>
        )}

        {queryHistory.length > 0 && (
          <div style={sectionStyle}>
            <h2 style={{ color: "#4338ca", fontSize: "24px", marginBottom: "16px" }}>Query History</h2>
            
            <select
              style={{...inputStyle, backgroundColor: "#f3f4f6"}}
              onChange={(e) => setQuery(e.target.value)}
            >
              <option value="">Select from history...</option>
              {queryHistory.map((q, index) => (
                <option key={index} value={q}>
                  {q}
                </option>
              ))}
            </select>
            
            <button 
              onClick={handleClearHistory} 
              style={{...buttonStyle, backgroundColor: "#dc2626", width: "100%"}}
            >
              Clear History
            </button>
          </div>
        )}

        <div style={sectionStyle}>
          <h2 style={{ color: "#4338ca", fontSize: "24px", marginBottom: "16px" }}>Query</h2>
          
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter query..."
            style={{...inputStyle, height: "100px", backgroundColor: "#f3f4f6"}}
          />
          
          <div style={{ display: "flex", justifyContent: "center", gap: "16px" }}>
            <button 
              onClick={handleQuery} 
              style={{...buttonStyle, backgroundColor: "#10b981", flex: 1}}
            >
              Get Insights
            </button>

            <button 
              onClick={handleCompare} 
              style={{...buttonStyle, backgroundColor: "#8b5cf6", flex: 1}}
            >
              Compare Reports
            </button>
          </div>
        </div>

        {response && (
          <div style={sectionStyle}>
            <h2 style={{ color: "#4338ca", fontSize: "24px", marginBottom: "16px" }}>Insights</h2>
            <div style={resultStyle}>
              <p>{response}</p>
            </div>
          </div>
        )}

        {comparisonResponse && (
          <div style={sectionStyle}>
            <h2 style={{ color: "#4338ca", fontSize: "24px", marginBottom: "16px" }}>Comparison Results</h2>
            <div style={{...resultStyle, backgroundColor: "#f5f3ff", border: "1px solid #ddd6fe"}}>
              <p>{comparisonResponse}</p>
            </div>
          </div>
        )}
      </div>
    </>
  );
}