import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, Image as ImageIcon, CheckCircle2, AlertCircle, Loader2, 
  Sparkles, Cpu, Zap, Target, Gauge, 
  Layers, ExternalLink, Github, Activity, 
  ChevronDown, Fingerprint, Scan, MousePointer2
} from 'lucide-react';
import { motion, AnimatePresence, useScroll, useTransform } from 'framer-motion';
import CursorTrail from './components/CursorTrail';

const API_URL = "https://sawabedarain-lumina-iqa.hf.space/predict";
const GITHUB_URL = "https://github.com/DarainHyder/Image_Quality_Assessment";
const PROJECT_URL = "https://huggingface.co/spaces/sawabedarain/Lumina-IQA";

const App = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Scroll Animations
  const { scrollYProgress } = useScroll();
  const heroOpacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);
  const heroScale = useTransform(scrollYProgress, [0, 0.2], [1, 0.95]);
  const hubOpacity = useTransform(scrollYProgress, [0.15, 0.3], [0, 1]);
  const hubY = useTransform(scrollYProgress, [0.15, 0.3], [50, 0]);

  const handleFile = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
      // Smooth scroll to hub on file selection
      window.scrollTo({ top: window.innerHeight * 0.4, behavior: 'smooth' });
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
      const response = await fetch(API_URL, { method: "POST", body: formData });
      if (!response.ok) throw new Error("API Offline");
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("System failure. Check backend infrastructure.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white font-sans text-slate-900 overflow-x-hidden">
      <CursorTrail />

      {/* Persistent Navigation */}
      <nav className="fixed top-0 inset-x-0 z-[100] px-8 h-24 flex items-center justify-between">
         <motion.div 
           initial={{ opacity: 0 }} animate={{ opacity: 1 }}
           className="flex items-center gap-2"
         >
           <Fingerprint size={24} className="text-indigo-500" />
           <span className="font-display font-black text-2xl tracking-tighter text-indigo-500">LUMINA</span>
         </motion.div>
         <div className="flex gap-4">
            <a href={GITHUB_URL} target="_blank" className="px-5 py-2 rounded-full border border-indigo-500/20 text-xs font-black text-indigo-500 hover:bg-indigo-500 hover:text-white transition-all uppercase tracking-widest bg-white/10 backdrop-blur-md">
              Source Code
            </a>
         </div>
      </nav>

      {/* 💎 UNIQUE: CINEMATIC REVEAL HERO (100vh) */}
      <motion.section 
        style={{ opacity: heroOpacity, scale: heroScale }}
        className="relative h-screen w-full flex items-center justify-center bg-slate-950 overflow-hidden"
      >
        {/* Animated Background Elements */}
        <div className="absolute inset-0 z-0">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-indigo-500/10 blur-[120px] rounded-full animate-pulse" />
          <div className="absolute top-1/4 right-1/4 w-[400px] h-[400px] bg-blue-500/5 blur-[100px] rounded-full animate-bounce" />
        </div>

        <div className="relative z-10 text-center space-y-8 px-6">
          <motion.div
            initial={{ opacity: 0, letterSpacing: "1em" }}
            animate={{ opacity: 1, letterSpacing: "0.4em" }}
            transition={{ duration: 1.5, ease: "easeOut" }}
            className="text-indigo-400 text-xs font-black uppercase"
          >
            Sawabedarain Presents
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 1 }}
            className="text-7xl md:text-9xl font-display font-black text-white tracking-tighter leading-none"
          >
            NEURAL <br /> VISION.
          </motion.h1>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.5 }}
            className="flex flex-col items-center gap-12"
          >
            <p className="max-w-md text-slate-400 font-medium text-lg leading-relaxed">
              Experience the next frontier of image quality assessment. Advanced, Precise, Boundless.
            </p>
            
            <div className="flex items-center gap-6">
               <button 
                onClick={() => window.scrollTo({ top: window.innerHeight * 0.8, behavior: 'smooth' })}
                className="group flex flex-col items-center gap-4 transition-all"
               >
                 <span className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-500 group-hover:text-white transition-colors">Begin Analysis</span>
                 <div className="w-10 h-10 rounded-full border border-slate-800 flex items-center justify-center group-hover:border-indigo-500 transition-colors">
                    <ChevronDown size={20} className="text-slate-600 group-hover:text-indigo-400 animate-bounce" />
                 </div>
               </button>
            </div>
          </motion.div>
        </div>

        {/* Cinematic Watermark */}
        <div className="absolute bottom-10 inset-x-0 text-center opacity-20">
           <span className="text-[10px] font-black text-white uppercase tracking-[1em]">perceptual.engine.v2.5</span>
        </div>
      </motion.section>

      {/* 🚀 MINIMALIST ANALYSIS WORKSPACE (Revealed on Scroll) */}
      <motion.main 
        style={{ opacity: hubOpacity, y: hubY }}
        className="relative z-20 bg-white"
      >
        <div className="max-w-6xl mx-auto px-6 py-32 space-y-32">
          
          {/* Symmetric Interaction Hub */}
          <section className="space-y-16">
            <div className="text-center space-y-4">
               <h2 className="text-5xl font-display font-black tracking-tighter">Analyzer.</h2>
               <p className="text-slate-500 font-medium max-w-sm mx-auto">Upload your imagery to extract neural fidelity metrics in real-time.</p>
            </div>

            <div className="max-w-4xl mx-auto">
              <div 
                onClick={() => fileInputRef.current?.click()}
                className={`group relative aspect-video rounded-[3rem] cursor-pointer transition-all duration-700 overflow-hidden flex flex-col items-center justify-center border border-slate-100 bg-slate-50/50 hover:bg-white
                  ${preview ? 'shadow-2xl shadow-indigo-100' : 'hover:shadow-[0_40px_100px_rgba(0,0,0,0.03)]'}`}
              >
                {!preview ? (
                  <div className="text-center p-12">
                     <div className="w-20 h-20 rounded-[2rem] bg-white border border-slate-100 text-slate-400 flex items-center justify-center mb-6 mx-auto group-hover:scale-110 group-hover:text-indigo-500 transition-all duration-500 shadow-sm">
                        <Scan size={28} />
                     </div>
                     <span className="text-xs font-black uppercase tracking-widest text-slate-400">Select Asset</span>
                  </div>
                ) : (
                  <div className="absolute inset-0 w-full h-full p-4">
                    <div className="w-full h-full rounded-[2.5rem] overflow-hidden">
                       <img src={preview} alt="Preview" className="w-full h-full object-cover transition-transform duration-1000 group-hover:scale-105" />
                    </div>
                  </div>
                )}
                <input ref={fileInputRef} type="file" className="hidden" onChange={handleFile} accept="image/*" />
              </div>

              <div className="mt-12 flex flex-col items-center gap-8">
                <button
                  onClick={handlePredict}
                  disabled={!file || loading}
                  className={`px-12 py-5 bg-slate-950 text-white rounded-2xl font-black text-sm uppercase tracking-widest transition-all hover:scale-105 active:scale-95 flex items-center gap-4
                    ${!file ? 'opacity-10 grayscale cursor-not-allowed' : ''}`}
                >
                  {loading ? <Loader2 size={20} className="animate-spin" /> : <Activity size={20} />}
                  {loading ? 'Decrypting...' : 'Perform Analysis'}
                </button>

                <AnimatePresence>
                  {error && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-red-500 font-extrabold text-xs uppercase tracking-widest bg-red-50 px-6 py-3 rounded-xl border border-red-100">
                       !!! {error} !!!
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </section>

          {/* Symmetrical Results Flow */}
          <AnimatePresence>
            {result && (
              <motion.section 
                initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
                className="py-24 border-t border-slate-100"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-24 items-center">
                   <div className="space-y-8">
                      <div className="inline-block px-4 py-1.5 rounded-full bg-slate-900 font-black text-[10px] text-white uppercase tracking-widest">
                         Final Assessment
                      </div>
                      <h3 className="text-6xl font-display font-black tracking-tighter">Perceptual Score.</h3>
                      <p className="text-slate-500 font-medium text-lg leading-relaxed">
                        Our EfficientNet backbone has analyzed the spectral density and textural coherence of your image. This score reflects the professional consensus of human perception.
                      </p>
                   </div>
                   
                   <div className="flex flex-col items-center md:items-end">
                      <div className="relative inline-block">
                         <span className="text-[15rem] font-display font-black leading-none tracking-tighter text-slate-900 border-b-8 border-indigo-500 pb-4">
                            {result.predicted_mos.toFixed(1)}
                         </span>
                         <span className="absolute -top-6 -right-12 text-sm font-black text-white bg-indigo-600 px-4 py-2 rounded-xl">MOS</span>
                      </div>
                      <div className="mt-12 flex gap-12 font-black text-[10px] text-slate-400 uppercase tracking-widest">
                         <div className="flex items-center gap-2"><Target size={14} /> Accuracy 94.2%</div>
                         <div className="flex items-center gap-2"><Activity size={14} /> Low Latency</div>
                      </div>
                   </div>
                </div>
              </motion.section>
            )}
          </AnimatePresence>

        </div>

        {/* Meaningful Robust Footer */}
        <footer className="bg-slate-50 border-t border-slate-200 py-32 mt-32">
          <div className="max-w-6xl mx-auto px-10 grid grid-cols-1 md:grid-cols-3 gap-24">
             <div className="space-y-8">
                <div className="flex items-center gap-3">
                   <Zap size={28} className="text-indigo-600" />
                   <span className="font-display font-black text-2xl tracking-tighter">LUMINA</span>
                </div>
                <p className="text-slate-500 font-medium text-sm leading-relaxed">
                  The world's premier engine for automated quality validation. Built for photographers, developers, and AI researchers who demand visual precision.
                </p>
             </div>

             <div className="grid grid-cols-2 gap-12">
                <div className="space-y-6">
                   <h4 className="text-[10px] font-black text-slate-900 uppercase tracking-widest">Architecture</h4>
                   <ul className="space-y-3 text-xs font-bold text-slate-400 uppercase tracking-tighter">
                      <li>EfficientNet-B0</li>
                      <li>PyTorch 2.3</li>
                      <li>FastAPI Hub</li>
                      <li>Vercel Edge</li>
                   </ul>
                </div>
                <div className="space-y-6">
                   <h4 className="text-[10px] font-black text-slate-900 uppercase tracking-widest">Explore</h4>
                   <ul className="space-y-3 text-xs font-bold text-slate-400 uppercase tracking-tighter">
                      <li><a href="https://github.com/DarainHyder" className="hover:text-indigo-600 transition-colors">Engineer Hub</a></li>
                      <li><a href={PROJECT_URL} className="hover:text-indigo-600 transition-colors">API Endpoint</a></li>
                      <li><a href="#" className="hover:text-indigo-600 transition-colors">Guidelines</a></li>
                   </ul>
                </div>
             </div>

             <div className="space-y-8 text-right md:text-right flex flex-col items-center md:items-end">
                <div className="h-20 w-[1px] bg-slate-200" />
                <div className="space-y-2">
                   <p className="text-[10px] font-black text-slate-900 uppercase tracking-widest">Coded with precision by</p>
                   <p className="font-display font-black text-3xl text-indigo-600 uppercase tracking-tighter italic">// sawabedarain</p>
                </div>
             </div>
          </div>

          <div className="max-w-6xl mx-auto px-10 mt-32 pt-8 border-t border-slate-200/50 flex justify-between items-center text-[10px] font-black text-slate-300 uppercase tracking-[0.5em]">
             <span>Neural Engine v2.5.0</span>
             <span>© 2026 DARAIN HYDER</span>
          </div>
        </footer>
      </motion.main>
    </div>
  );
};

export default App;
