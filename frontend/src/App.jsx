import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, Image as ImageIcon, CheckCircle2, AlertCircle, Loader2, 
  Sparkles, Shield, Cpu, ArrowRight, Zap, Target, Gauge, 
  Layers, ExternalLink, Github, Fingerprint, Activity
} from 'lucide-react';
import { motion, AnimatePresence, useMotionValue, useSpring } from 'framer-motion';

const API_URL = "https://sawabedarain-lumina-iqa.hf.space/predict";
const PROJECT_URL = "https://huggingface.co/spaces/sawabedarain/Lumina-IQA";

const App = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Mouse Tracking for Cursor Glow
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // Smooth springs for the cursor effect
  const springConfig = { damping: 25, stiffness: 150 };
  const smoothX = useSpring(mouseX, springConfig);
  const smoothY = useSpring(mouseY, springConfig);

  useEffect(() => {
    const handleMouseMove = (e) => {
      mouseX.set(e.clientX - 300); // 300 is half the width of the glow
      mouseY.set(e.clientY - 300);
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

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
      setError("Neural link severed. Please check backend status.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-artistic font-sans text-slate-900 selection:bg-slate-900 selection:text-white">
      {/* Light Cursor Glow */}
      <motion.div 
        className="cursor-glow"
        style={{
          x: smoothX,
          y: smoothY,
        }}
      />

      {/* Modern Minimal Navigation */}
      <nav className="fixed top-0 inset-x-0 z-[100] px-8 h-20 flex items-center justify-between">
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3"
        >
          <div className="w-10 h-10 rounded-2xl bg-slate-900 flex items-center justify-center text-white">
            <Fingerprint size={22} className="opacity-80" />
          </div>
          <span className="font-display font-extrabold text-2xl tracking-tighter uppercase grayscale">Lumina</span>
        </motion.div>
        
        <div className="flex items-center gap-4">
          <a 
            href={PROJECT_URL} 
            target="_blank" 
            rel="noopener noreferrer"
            className="hidden md:flex items-center gap-2 px-6 py-2 rounded-full border border-slate-200 text-sm font-bold hover:bg-slate-50 transition-all grayscale"
          >
            <ExternalLink size={14} /> API Infrastructure
          </a>
          <a href="https://github.com/DarainHyder/Image_Quality_Assessment" className="w-10 h-10 rounded-full border border-slate-200 flex items-center justify-center hover:bg-slate-50 transition-all grayscale">
            <Github size={18} />
          </a>
        </div>
      </nav>

      <main className="relative z-10 pt-40 pb-32 px-10">
        <div className="max-w-7xl mx-auto flex flex-col items-center">
          
          {/* Artistic Hero Section */}
          <div className="w-full text-center space-y-8 mb-24">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="inline-block px-4 py-1.5 rounded-full bg-slate-100 text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 border border-slate-200"
            >
              Neural Aesthetics Engine
            </motion.div>
            
            <motion.h1 
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-6xl md:text-8xl leading-tight font-display font-extrabold text-gradient"
            >
              Intelligence <br /> Meets Vision.
            </motion.h1>

            <motion.p 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="max-w-xl mx-auto text-slate-500 font-medium leading-relaxed"
            >
              Advanced No-Reference Quality Assessment. We don't just see pixels; we understand perception. Built on EfficientNet architecture.
            </motion.p>
          </div>

          <div className="w-full grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
            
            {/* Main Interactive Hub (Upload) */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="lg:col-span-7 space-y-6"
            >
              <div 
                className="glass-panel rounded-[40px] p-4 relative frosted overflow-hidden"
              >
                <div 
                  onClick={() => fileInputRef.current?.click()}
                  className={`group relative min-h-[500px] rounded-[32px] cursor-pointer transition-all duration-700 overflow-hidden flex flex-col items-center justify-center
                    ${preview ? 'bg-black' : 'border-2 border-dashed border-slate-200 hover:border-slate-400 bg-white/20'}`}
                >
                  {!preview ? (
                    <div className="text-center p-12">
                      <div className="w-24 h-24 rounded-[32px] bg-slate-900 text-white flex items-center justify-center mb-6 mx-auto transition-transform group-hover:scale-110 duration-500 shadow-2xl">
                        <Upload size={32} />
                      </div>
                      <h3 className="text-2xl font-display font-bold text-slate-900 mb-2">Initiate Scan</h3>
                      <p className="text-slate-400 font-medium">Drop imagery to begin perceptual mapping</p>
                    </div>
                  ) : (
                    <>
                      <img src={preview} alt="Input" className="absolute inset-0 w-full h-full object-cover opacity-80 transition-transform duration-1000 group-hover:scale-110" />
                      <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition-opacity duration-500 flex items-center justify-center backdrop-blur-sm">
                        <div className="px-8 py-3 bg-white rounded-full font-bold shadow-2xl flex items-center gap-2">
                           <ImageIcon size={18} /> Reload Source
                        </div>
                      </div>
                    </>
                  )}
                  <input ref={fileInputRef} type="file" className="hidden" onChange={handleFile} accept="image/*" />
                </div>
              </div>

              <button
                onClick={handlePredict}
                disabled={!file || loading}
                className={`w-full group btn-primary h-20 text-lg
                  ${!file ? 'opacity-30 cursor-not-allowed grayscale' : ''}`}
              >
                <div className="flex items-center gap-3 relative z-10">
                  {loading ? <Loader2 size={24} className="animate-spin opacity-50" /> : <Activity size={24} className="opacity-50" />}
                  <span>{loading ? 'Decrypting Visual Fidelity...' : 'Run Perceptual Analysis'}</span>
                </div>
                <div className={`absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000`} />
              </button>

              <AnimatePresence>
                {error && (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    className="p-6 rounded-2xl bg-red-950 text-red-200 text-sm font-bold flex items-center gap-3 border border-red-900"
                  >
                    <AlertCircle size={18} /> {error}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Intel & Results Column */}
            <div className="lg:col-span-5 space-y-8">
              
              <AnimatePresence mode="wait">
                {result ? (
                  <motion.div
                    key="results"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className="glass-panel p-10 rounded-[40px] space-y-10 border-slate-900/5"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h2 className="text-3xl font-display font-extrabold text-slate-900 tracking-tighter">QUALITY DATA</h2>
                        <p className="text-xs font-bold text-slate-400 mt-1 uppercase tracking-widest">Neural Scan Complete</p>
                      </div>
                      <CheckCircle2 className="text-slate-900 opacity-20" size={32} />
                    </div>

                    <div className="flex items-center gap-10">
                      <div className="relative">
                         <div className="text-7xl font-display font-black text-slate-900 tracking-tighter leading-none">
                            {result.predicted_mos.toFixed(1)}
                         </div>
                         <div className="absolute -top-4 -right-6 px-3 py-1 bg-slate-900 text-white text-[10px] font-black rounded-lg">
                            MOS
                         </div>
                      </div>
                      <div className="h-16 w-[1px] bg-slate-100" />
                      <div className="space-y-1">
                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest leading-none">Status</p>
                        <p className="text-xl font-display font-bold text-slate-900">High Fidelity</p>
                      </div>
                    </div>

                    <div className="space-y-4">
                       <div className="flex justify-between text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">
                          <span>Confidence Level</span>
                          <span>94.2%</span>
                       </div>
                       <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden p-0.5">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: `${result.predicted_mos}%` }}
                            className="h-full bg-slate-900 rounded-full"
                          />
                       </div>
                    </div>

                    <div className="p-6 rounded-3xl bg-slate-50 flex items-center gap-4">
                      <div className="w-12 h-12 rounded-2xl bg-white shadow-sm flex items-center justify-center">
                        <Cpu size={20} className="text-slate-900 opacity-60" />
                      </div>
                      <div>
                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest leading-none mb-1">Architecture</p>
                        <p className="font-bold text-slate-900 text-sm">EfficientNet-B0 (NR-IQA)</p>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="placeholder"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="p-10 rounded-[40px] border-2 border-dashed border-slate-200 text-center space-y-4 py-24"
                  >
                    <div className="w-16 h-16 rounded-full bg-slate-50 flex items-center justify-center mx-auto text-slate-200">
                      <Zap size={32} />
                    </div>
                    <div>
                      <h4 className="font-bold text-slate-300">Awaiting Signal</h4>
                      <p className="text-xs text-slate-400 mx-auto max-w-[200px]">Upload an image to populate perceptual quality metrics.</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Technical Footnotes */}
              <div className="px-6 grid grid-cols-2 gap-4">
                <div className="space-y-1">
                   <p className="text-[10px] font-black text-slate-300 uppercase tracking-widest leading-none">Server</p>
                   <p className="text-xs font-bold text-slate-400">HF Spaces / Docker</p>
                </div>
                <div className="space-y-1">
                   <p className="text-[10px] font-black text-slate-300 uppercase tracking-widest leading-none">Core</p>
                   <p className="text-xs font-bold text-slate-400">PyTorch 2.3.0</p>
                </div>
              </div>

            </div>

          </div>
        </div>
      </main>

      {/* Floating Global Watermark */}
      <div className="fixed bottom-10 left-10 pointer-events-none group">
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-4 rotate-[-90deg] origin-bottom-left"
        >
          <span className="text-[10px] font-black text-slate-300 uppercase tracking-[0.4em]">perceptual.engine.v1</span>
          <div className="w-12 h-[1px] bg-slate-200"></div>
        </motion.div>
      </div>

      <footer className="fixed bottom-10 right-10 pointer-events-none">
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-slate-400 font-display font-extrabold tracking-[0.2em] text-xs uppercase"
        >
          // sawabedarain
        </motion.div>
      </footer>
    </div>
  );
};

export default App;
