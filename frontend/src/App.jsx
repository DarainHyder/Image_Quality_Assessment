import React, { useState, useRef } from 'react';
import { 
  Upload, Image as ImageIcon, CheckCircle2, AlertCircle, Loader2, 
  Sparkles, Shield, Cpu, ArrowRight, Zap, Target, Gauge, 
  Layers, ExternalLink, Github
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_URL = "https://sawabedarain-lumina-iqa.hf.space/predict";

const App = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFile = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("API Connection failed");

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Analysis failed. Please check the backend connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-mesh-gradient font-sans text-slate-800 selection:bg-lumina-100">
      {/* Navigation */}
      <nav className="fixed top-0 inset-x-0 z-50 bg-white/50 backdrop-blur-md border-b border-slate-200/50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-2"
          >
            <div className="w-8 h-8 rounded-lg bg-lumina-600 flex items-center justify-center text-white font-bold text-lg">L</div>
            <span className="font-display font-bold text-xl tracking-tight text-slate-900">Lumina IQA</span>
          </motion.div>
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-600">
            <a href="#" className="hover:text-lumina-600 transition-colors">Technology</a>
            <a href="#" className="hover:text-lumina-600 transition-colors">Benchmarks</a>
            <a href="#" className="hover:text-lumina-600 transition-colors">API Docs</a>
            <a href="https://github.com/DarainHyder/Image_Quality_Assessment" className="btn-secondary py-2 px-4 flex items-center gap-2 text-sm">
              <Github size={16} /> GitHub
            </a>
          </div>
        </div>
      </nav>

      <main className="pt-32 pb-24 px-6">
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 items-start">
          
          {/* Left Column: Hero & Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-lumina-50 text-lumina-700 text-xs font-bold uppercase tracking-wider mb-6 border border-lumina-100">
              <Sparkles size={14} /> EfficientNet-B0 Model
            </div>
            <h1 className="text-5xl md:text-6xl font-display font-bold leading-[1.1] mb-6 text-slate-900">
              Precise Insight <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-lumina-600 to-indigo-500">
                Into Every Pixel.
              </span>
            </h1>
            <p className="text-lg text-slate-600 leading-relaxed mb-8 max-w-lg">
              Assess the perceptual quality of any image instantly using our state-of-the-art No-Reference Quality Assessment engine. Trained on KonIQ-10k.
            </p>

            <div className="grid grid-cols-2 gap-6 mb-12">
              <div className="p-4 rounded-2xl bg-white/50 border border-slate-200">
                <div className="w-10 h-10 rounded-xl bg-lumina-100 text-lumina-600 flex items-center justify-center mb-3">
                  <Cpu size={20} />
                </div>
                <h3 className="font-bold mb-1">Deep Intelligence</h3>
                <p className="text-xs text-slate-500">EfficientNet-B0 backbone for superior accuracy.</p>
              </div>
              <div className="p-4 rounded-2xl bg-white/50 border border-slate-200">
                <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-600 flex items-center justify-center mb-3">
                  <Zap size={20} />
                </div>
                <h3 className="font-bold mb-1">Real-time Inference</h3>
                <p className="text-xs text-slate-500">Optimized for low-latency scoring on Docker.</p>
              </div>
            </div>

            <div className="flex items-center gap-12 text-slate-400 font-bold tracking-widest text-[10px] uppercase">
              <span>PYTORCH</span>
              <span>FASTAPI</span>
              <span>REACT.JS</span>
              <span>KAGGLE</span>
            </div>
          </motion.div>

          {/* Right Column: Interaction Zone */}
          <div className="space-y-6">
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="glass-card p-8 rounded-[32px] relative overflow-hidden"
            >
              {/* Image Upload Area */}
              <div 
                onClick={() => fileInputRef.current?.click()}
                className={`relative group cursor-pointer border-2 border-dashed rounded-2xl transition-all duration-500 min-h-[300px] flex flex-col items-center justify-center overflow-hidden
                  ${preview ? 'border-transparent' : 'border-slate-300 hover:border-lumina-400 hover:bg-lumina-50/30'}`}
              >
                {!preview ? (
                  <div className="text-center p-8">
                    <div className="w-16 h-16 rounded-2xl bg-slate-50 text-slate-400 flex items-center justify-center mb-4 mx-auto group-hover:scale-110 group-hover:bg-lumina-100 group-hover:text-lumina-600 transition-all duration-300">
                      <Upload size={32} />
                    </div>
                    <p className="font-bold text-slate-900 mb-1">Upload an Image</p>
                    <p className="text-sm text-slate-500">Drag and drop or click to browse</p>
                  </div>
                ) : (
                  <>
                    <img src={preview} alt="Preview" className="absolute inset-0 w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                      <p className="text-white font-bold flex items-center gap-2">
                        <ImageIcon size={20} /> Replace Image
                      </p>
                    </div>
                  </>
                )}
                <input ref={fileInputRef} type="file" className="hidden" onChange={handleFile} accept="image/*" />
              </div>

              {/* Action Button */}
              <button
                onClick={handlePredict}
                disabled={!file || loading}
                className={`w-full mt-6 flex items-center justify-center gap-3 py-4 rounded-xl font-bold transition-all duration-300
                  ${!file ? 'bg-slate-100 text-slate-400 cursor-not-allowed' : 
                    loading ? 'bg-lumina-100 text-lumina-500' : 'btn-primary'}`}
              >
                {loading ? <Loader2 className="animate-spin" /> : <Target size={20} />}
                {loading ? 'Analyzing Neural Patterns...' : 'Analyze Quality'}
              </button>

              {/* Status & Errors */}
              <AnimatePresence>
                {error && (
                  <motion.div 
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="mt-6 p-4 rounded-xl bg-red-50 text-red-600 flex items-center gap-3 text-sm font-medium border border-red-100"
                  >
                    <AlertCircle size={18} />
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Analysis Results Panel */}
            <AnimatePresence>
              {result && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-8 rounded-[32px]"
                >
                  <div className="flex items-center justify-between mb-8">
                    <h2 className="flex items-center gap-2 font-display font-bold text-xl">
                      <Gauge className="text-lumina-600" size={24} /> Quality Report
                    </h2>
                    <div className="px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 text-[10px] font-bold uppercase tracking-widest flex items-center gap-1.5 border border-emerald-100">
                      <CheckCircle2 size={12} /> Analysis Complete
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                    <div className="relative flex flex-col items-center">
                      <div className="relative w-40 h-40">
                        {/* Circle Progress Bar Backdrop */}
                        <svg className="w-full h-full transform -rotate-90">
                          <circle cx="80" cy="80" r="70" stroke="currentColor" strokeWidth="12" fill="transparent" className="text-slate-100" />
                          <motion.circle 
                            cx="80" cy="80" r="70" 
                            stroke="currentColor" strokeWidth="12" fill="transparent" 
                            className="text-lumina-500"
                            strokeDasharray={440}
                            initial={{ strokeDashoffset: 440 }}
                            animate={{ strokeDashoffset: 440 - (440 * (result.predicted_mos / 100)) }}
                            transition={{ duration: 1.5, ease: "easeOut" }}
                          />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                          <span className="text-4xl font-display font-bold text-slate-900">
                            {result.predicted_mos.toFixed(1)}
                          </span>
                          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                            MOS Score
                          </span>
                        </div>
                      </div>
                      <p className="mt-4 text-xs font-medium text-slate-500 text-center">
                        Mean Opinion Score based on <br /> per-pixel feature mapping.
                      </p>
                    </div>

                    <div className="space-y-6">
                      <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100">
                        <div className="flex justify-between text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">
                          <span>Quality Grade</span>
                          <span className="text-lumina-600">Excellent</span>
                        </div>
                        <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: `${result.predicted_mos}%` }}
                            className="h-full bg-gradient-to-r from-lumina-400 to-lumina-600"
                          />
                        </div>
                      </div>
                      
                      <div className="p-4 rounded-2xl bg-indigo-50/50 border border-indigo-100/50">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-lg bg-indigo-100 text-indigo-600 flex items-center justify-center">
                            <Layers size={16} />
                          </div>
                          <div>
                            <p className="text-[10px] font-bold text-indigo-900/50 uppercase tracking-widest leading-none mb-1">Model Backbone</p>
                            <p className="font-bold text-indigo-900 text-sm">EfficientNet-B0 (Headless)</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>

      {/* Footer / Watermark */}
      <footer className="fixed bottom-6 right-8 pointer-events-none">
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="flex items-center gap-3 text-slate-300 font-display font-bold tracking-[0.2em] text-[10px] uppercase"
        >
          <span className="w-8 h-[1px] bg-slate-200"></span>
          sawabedarain
        </motion.div>
      </footer>
    </div>
  );
};

export default App;
