import React, { useState, useCallback } from 'react';
import { Upload, Image as ImageIcon, CheckCircle2, AlertCircle, Loader2, Sparkles, Shield, Cpu } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_URL = "https://YOUR_HF_SPACE_URL.hf.space/predict"; // Replace with actual HF Space URL after deployment

const App = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      handleFile(selectedFile);
    }
  };

  const handleFile = (selectedFile) => {
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const analyzeImage = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Failed to analyze image');
      const data = await response.json();
      setResult(data.quality_score);
    } catch (err) {
      setError("Prediction service unavailable. Ensure backend is deployed.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pb-20">
      {/* Navigation */}
      <nav className="container py-8 flex justify-between items-center bg-white/50 backdrop-blur-sm sticky top-0 z-50 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white shadow-lg shadow-blue-200">
            <ImageIcon size={22} />
          </div>
          <span className="text-xl font-bold font-display tracking-tight text-slate-900">
            VisionQuality<span className="text-blue-600">.</span>
          </span>
        </div>
        <div className="hidden md:flex gap-8 text-sm font-medium text-slate-600">
          <a href="#" className="hover:text-blue-600 transition-colors">Technology</a>
          <a href="#" className="hover:text-blue-600 transition-colors">Benchmarks</a>
          <a href="#" className="hover:text-blue-600 transition-colors">Documentation</a>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="container pt-20 pb-16 text-center max-w-4xl mx-auto">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 text-blue-700 text-xs font-bold uppercase tracking-wider mb-6">
            <Sparkles size={14} />
            Deep Learning NR-IQA
          </div>
          <h1 className="text-5xl md:text-7xl mb-6 bg-clip-text text-transparent bg-gradient-to-br from-slate-900 via-slate-800 to-slate-600">
            Precise Insight Into Every Pixel.
          </h1>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
            Assess the perceptual quality of any image instantly using our state-of-the-art 
            No-Reference Quality Assessment engine built with EfficientNet-B0.
          </p>
        </motion.div>
      </header>

      {/* Main Action Area */}
      <main className="container max-w-5xl">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          
          {/* Upload Card */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="premium-card p-2 group"
          >
            <div className={`
              border-2 border-dashed rounded-[20px] p-8 h-[400px] flex flex-col items-center justify-center transition-all duration-300
              ${preview ? 'border-transparent p-0 overflow-hidden' : 'border-slate-200 group-hover:border-blue-300 group-hover:bg-blue-50/50'}
            `}>
              {preview ? (
                <div className="relative w-full h-full group">
                  <img src={preview} alt="Preview" className="w-full h-full object-cover rounded-[18px]" />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-[2px]">
                    <label className="btn-primary cursor-pointer">
                      Change Image
                      <input type="file" className="hidden" onChange={onFileChange} accept="image/*" />
                    </label>
                  </div>
                </div>
              ) : (
                <label className="flex flex-col items-center cursor-pointer w-full h-full justify-center">
                  <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center text-blue-600 mb-4 group-hover:scale-110 transition-transform duration-300">
                    <Upload size={32} />
                  </div>
                  <p className="font-display font-semibold text-slate-900 mb-1 text-lg">Upload an Image</p>
                  <p className="text-sm text-slate-500">Drag and drop or click to browse</p>
                  <input type="file" className="hidden" onChange={onFileChange} accept="image/*" />
                </label>
              )}
            </div>
          </motion.div>

          {/* Results Card */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="premium-card p-10 flex flex-col min-h-[400px]"
          >
            <div className="flex-grow">
              <h3 className="text-2xl mb-8 flex items-center gap-3">
                <Shield className="text-blue-600" size={24} />
                Quality Report
              </h3>

              {!preview && !result && (
                <div className="flex flex-col items-center justify-center h-48 text-slate-400">
                  <ImageIcon size={48} className="mb-4 opacity-20" />
                  <p className="text-sm font-medium">Capture or upload an image to start analysis</p>
                </div>
              )}

              {preview && !result && !loading && (
                <div className="space-y-6">
                  <p className="text-slate-600 text-sm leading-relaxed">
                    Image ready for processing. Our AI will analyze artifacts, blur, and perceptual coherence.
                  </p>
                  <button 
                    onClick={analyzeImage}
                    className="btn-primary w-full py-4 text-lg justify-center shadow-lg shadow-blue-200"
                  >
                    Analyze Quality
                  </button>
                </div>
              )}

              {loading && (
                <div className="flex flex-col items-center justify-center h-48 py-8 animate-pulse">
                  <Loader2 size={40} className="text-blue-600 animate-spin mb-4" />
                  <p className="font-display font-medium text-slate-800">Processing Pixels...</p>
                </div>
              )}

              {result !== null && (
                <div className="animate-fade-in space-y-8">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Perceptual Score</p>
                      <h4 className="text-6xl font-display font-black text-slate-900">
                        {result}
                      </h4>
                    </div>
                    <div className="w-16 h-16 bg-green-50 text-green-600 rounded-full flex items-center justify-center">
                      <CheckCircle2 size={32} />
                    </div>
                  </div>

                  <div className="w-full bg-slate-100 h-3 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(result, 100)}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className="h-full bg-gradient-to-r from-blue-600 to-indigo-600"
                    />
                  </div>

                  <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 flex gap-4">
                    <div className="bg-white p-2 rounded-lg shadow-sm h-fit">
                      <Cpu size={18} className="text-slate-600" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-slate-800 mb-1">EfficientNet Analysis</p>
                      <p className="text-xs text-slate-500 leading-relaxed">
                        The score indicates the Mean Opinion Score (MOS) predicted based on the KonIQ-10k dataset standards.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {error && (
                <div className="p-4 bg-red-50 text-red-700 rounded-xl border border-red-100 flex items-start gap-3 mt-4">
                  <AlertCircle size={20} className="mt-0.5 shrink-0" />
                  <p className="text-sm font-medium">{error}</p>
                </div>
              )}
            </div>
            
            {result && (
              <button 
                onClick={() => {setResult(null); setPreview(null); setFile(null);}} 
                className="mt-8 text-sm font-semibold text-slate-400 hover:text-blue-600 transition-colors self-center underline underline-offset-4"
              >
                Start New Assessment
              </button>
            )}
          </motion.div>
        </div>
      </main>

      {/* Footer Info */}
      <footer className="container mt-24 text-center border-t border-slate-100 pt-12">
        <p className="text-xs font-bold text-slate-400 tracking-widest uppercase mb-4">Enterprise Grade Intelligence</p>
        <div className="flex flex-wrap justify-center gap-12 opacity-50 grayscale hover:grayscale-0 transition-all duration-700">
           <span className="text-xl font-black font-display italic">PYTORCH</span>
           <span className="text-xl font-black font-display italic">FASTAPI</span>
           <span className="text-xl font-black font-display italic">REACT.JS</span>
           <span className="text-xl font-black font-display italic">KAGGLE</span>
        </div>
      </footer>

      {/* Watermark as requested */}
      <div className="watermark">
        sawabedarain
      </div>
    </div>
  );
};

export default App;
