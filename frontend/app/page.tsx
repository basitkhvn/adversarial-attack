"use client";

import { useState, useRef } from "react";
import { Upload, AlertTriangle, ShieldCheck, ShieldAlert, Zap, RefreshCw, ArrowRight } from "lucide-react";

export default function Home() {
  // --- STATE ---
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [epsilon, setEpsilon] = useState<number>(0.1);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- HANDLERS ---
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected)); // Show preview immediately
      setResult(null); // Reset previous results
    }
  };

  const handleAttack = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("epsilon", epsilon.toString());

    try {
      // Connect to Python Backend
      const response = await fetch("http://127.0.0.1:8000/attack", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Error connecting to backend. Is it running on port 8000?");
    } finally {
      setLoading(false);
    }
  };

  // --- UI COMPONENTS ---
  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 font-sans selection:bg-indigo-500 selection:text-white">
      
      {/* HEADER */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center gap-3">
          <div className="p-2 bg-indigo-600 rounded-lg">
            <ShieldAlert className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">FGSM Attack Lab</h1>
            <p className="text-xs text-slate-400 font-mono">ADVERSARIAL ROBUSTNESS DEMO</p>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-10 grid md:grid-cols-12 gap-10">
        
        {/* LEFT COLUMN: CONTROLS */}
        <div className="md:col-span-5 space-y-8">
          
          {/* 1. Upload Section */}
          <div className="bg-slate-800 rounded-xl p-1 shadow-lg border border-slate-700">
            <div 
              className={`border-2 border-dashed rounded-lg p-8 flex flex-col items-center justify-center text-center transition-all cursor-pointer relative h-64
                ${preview ? "border-indigo-500/50 bg-slate-900" : "border-slate-600 hover:border-slate-500 hover:bg-slate-700/50"}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                ref={fileInputRef}
                type="file" 
                accept="image/*" 
                className="hidden" 
                onChange={handleFileChange}
              />
              
              {preview ? (
                <img 
                  src={preview} 
                  alt="Upload Preview" 
                  className="h-full w-full object-contain rounded" 
                />
              ) : (
                <>
                  <div className="bg-slate-700 p-4 rounded-full mb-4">
                    <Upload className="w-8 h-8 text-slate-400" />
                  </div>
                  <p className="font-medium text-slate-300">Click to Upload Digit</p>
                  <p className="text-xs text-slate-500 mt-2">Supports PNG, JPG (MNIST Style)</p>
                </>
              )}
            </div>
          </div>

          {/* 2. Parameters Section */}
          <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700">
            <div className="flex items-center justify-between mb-4">
              <label className="flex items-center gap-2 font-bold text-white">
                <Zap className="w-4 h-4 text-yellow-400" />
                Attack Strength (ε)
              </label>
              <span className="font-mono text-indigo-400 bg-indigo-400/10 px-2 py-1 rounded text-sm">
                {epsilon.toFixed(2)}
              </span>
            </div>
            
            <input 
              type="range" 
              min="0" max="0.5" step="0.01"
              value={epsilon}
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400"
            />
            <div className="flex justify-between text-xs text-slate-500 mt-2 font-mono">
              <span>0.00 (No Noise)</span>
              <span>0.50 (High Noise)</span>
            </div>
          </div>

          {/* 3. Action Button */}
          <button
            onClick={handleAttack}
            disabled={!file || loading}
            className={`w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-2 transition-all shadow-lg
              ${!file 
                ? "bg-slate-700 text-slate-500 cursor-not-allowed" 
                : loading 
                  ? "bg-indigo-600/80 cursor-wait" 
                  : "bg-indigo-600 hover:bg-indigo-500 hover:scale-[1.02] text-white hover:shadow-indigo-500/25"
              }`}
          >
            {loading ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin" /> Generating Adversarial Example...
              </>
            ) : (
              <>
                Run FGSM Attack <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>
        </div>

        {/* RIGHT COLUMN: RESULTS */}
        <div className="md:col-span-7">
          {!result ? (
            // Placeholder State
            <div className="h-full border-2 border-dashed border-slate-700 rounded-xl flex flex-col items-center justify-center text-slate-500 min-h-[400px]">
              <AlertTriangle className="w-12 h-12 mb-4 opacity-20" />
              <p>Upload an image and run the attack to see results.</p>
            </div>
          ) : (
            // Result State
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              
              {/* Status Banner */}
              <div className={`p-4 rounded-xl border flex items-center gap-4 shadow-lg
                ${result.status === "Success" 
                  ? "bg-emerald-500/10 border-emerald-500/50 text-emerald-400" 
                  : "bg-red-500/10 border-red-500/50 text-red-400"
                }`}>
                {result.status === "Success" ? (
                  <ShieldCheck className="w-8 h-8" />
                ) : (
                  <ShieldAlert className="w-8 h-8" />
                )}
                <div>
                  <h3 className="font-bold text-lg">
                    {result.status === "Success" ? "Attack Successful" : "Attack Failed"}
                  </h3>
                  <p className="text-sm opacity-80">
                    {result.status === "Success" 
                      ? "The model was fooled into misclassifying the image." 
                      : "The model correctly identified the image despite the noise."}
                  </p>
                </div>
              </div>

              {/* Comparison Grid */}
              <div className="grid grid-cols-2 gap-6">
                
                {/* CLEAN CARD */}
                <div className="bg-slate-800 rounded-xl overflow-hidden border border-slate-700 shadow-lg">
                  <div className="bg-slate-900/50 p-3 border-b border-slate-700 text-center">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Original Input</span>
                  </div>
                  <div className="p-6 flex flex-col items-center">
                    <div className="relative w-32 h-32 mb-4">
                       {/* Display Original Image */}
                       {preview && <img src={preview} className="w-full h-full object-contain drop-shadow-lg" />}
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-slate-500 uppercase font-bold mb-1">Prediction</p>
                      <p className="text-4xl font-black text-emerald-400">{result.original_prediction}</p>
                    </div>
                  </div>
                </div>

                {/* ADVERSARIAL CARD */}
                <div className="bg-slate-800 rounded-xl overflow-hidden border border-red-500/30 shadow-lg relative">
                  <div className="absolute top-0 right-0 bg-red-500 text-white text-[10px] font-bold px-2 py-1 rounded-bl-lg">
                    NOISE ADDED
                  </div>
                  <div className="bg-slate-900/50 p-3 border-b border-slate-700 text-center">
                    <span className="text-xs font-bold uppercase tracking-wider text-red-300">Adversarial Input</span>
                  </div>
                  <div className="p-6 flex flex-col items-center">
                    <div className="relative w-32 h-32 mb-4">
                      {/* Display Base64 Image from Backend */}
                      <img 
                        src={`data:image/png;base64,${result.image}`} 
                        className="w-full h-full object-contain drop-shadow-lg filter contrast-125" 
                      />
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-slate-500 uppercase font-bold mb-1">Prediction</p>
                      <p className="text-4xl font-black text-red-400">{result.adversarial_prediction}</p>
                    </div>
                  </div>
                </div>

              </div>

              {/* JSON Logs (Optional, purely for 'Hacker' aesthetic) */}
              <div className="bg-black/40 rounded-lg p-4 font-mono text-xs text-slate-500 overflow-x-auto border border-slate-800">
                <p className="mb-2 text-slate-400 font-bold">// SYSTEM LOGS</p>
                <p>{`> Input Shape: [1, 1, 28, 28]`}</p>
                <p>{`> Gradient Calculation: Complete`}</p>
                <p>{`> Perturbation (ε=${epsilon}): Applied`}</p>
                <p className={result.status === "Success" ? "text-emerald-500" : "text-red-500"}>
                  {`> Result: ${result.status === "Success" ? "MISCLASSIFICATION_CONFIRMED" : "ROBUSTNESS_VERIFIED"}`}
                </p>
              </div>

            </div>
          )}
        </div>
      </main>
    </div>
  );
}