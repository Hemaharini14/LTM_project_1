import { motion } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import { MagneticButton } from './ui';

export function Hero() {
  return (
    <section id="hero" className="relative flex min-h-screen items-center">
      <div className="mx-auto w-full max-w-[1500px] px-6 sm:px-10">
        <div className="max-w-[46rem]">
          <motion.div
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
            className="flex items-center gap-3"
          >
            <span className="h-px w-8 bg-cyan/50" />
            <span className="label">AI-Powered Aviation Intelligence</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.2, delay: 0.25, ease: [0.22, 1, 0.36, 1] }}
            className="display mt-7 text-[clamp(3.2rem,11vw,9.5rem)] text-white"
          >
            PREDICT<br />
            <span className="text-white/35">THE JOURNEY.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.45, ease: [0.22, 1, 0.36, 1] }}
            className="mt-8 max-w-md text-[15px] leading-relaxed text-white/55"
          >
            Predict flight delays, understand why they happen, and build smarter trips with AI.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="mt-11 flex flex-wrap gap-3"
          >
            <MagneticButton href="/flight-delay">Explore Flight Intelligence</MagneticButton>
            <MagneticButton href="/budget-trip" variant="ghost">Plan A Trip</MagneticButton>
          </motion.div>
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.4, duration: 1 }}
        className="absolute bottom-9 left-1/2 flex -translate-x-1/2 flex-col items-center gap-2"
      >
        <span className="font-mono text-[9px] uppercase tracking-[0.3em] text-white/30">Scroll to explore</span>
        <ChevronDown className="h-3.5 w-3.5 animate-bounce text-cyan/60" strokeWidth={1.5} />
      </motion.div>
    </section>
  );
}
