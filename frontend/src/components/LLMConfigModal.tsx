import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  X,
  Cpu,
  Cloud,
  ShieldCheck,
  Zap,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Radio,
} from 'lucide-react'
import { llmApi } from '../api/client'
import type { LLMTestResponse } from '../types'

interface LLMConfigModalProps {
  open: boolean
  onClose: () => void
}

export function LLMConfigModal({ open, onClose }: LLMConfigModalProps) {
  const qc = useQueryClient()
  const [testResult, setTestResult] = useState<LLMTestResponse | null>(null)

  const statusQ = useQuery({
    queryKey: ['llm-status'],
    queryFn: llmApi.getStatus,
    enabled: open,
  })

  const updateCfg = useMutation({
    mutationFn: llmApi.configure,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      setTestResult(null)
    },
  })

  const pingLLM = useMutation({
    mutationFn: llmApi.testConnection,
    onSuccess: (res) => {
      setTestResult(res)
    },
  })

  if (!open) return null

  const st = statusQ.data

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md p-4">
      <div className="w-full max-w-lg max-h-[88vh] flex flex-col rounded-3xl glass-panel border border-white/15 shadow-[0_24px_80px_rgba(0,0,0,0.85)] overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-[#e5b869] to-[#5ec2a8] p-[1px]">
              <div className="w-full h-full bg-[#080a0a] rounded-[11px] flex items-center justify-center">
                <Radio className="w-4 h-4 text-[#e5b869]" />
              </div>
            </div>
            <div>
              <h3 className="font-brand font-bold text-base text-[#f3efe6]">
                Neural Language Model Router
              </h3>
              <p className="text-[11px] font-mono text-zinc-400">
                Switch between 100% Local Ollama and Cloud Inference
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-full text-zinc-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-5 text-xs overflow-y-auto">
          {statusQ.isLoading && (
            <div className="flex items-center gap-2 text-zinc-400 py-6 justify-center">
              <Loader2 className="w-4 h-4 animate-spin" /> Inspecting LLM providers…
            </div>
          )}

          {st && (
            <>
              {/* Privacy & Routing Status Banner */}
              <div
                className={`p-3.5 rounded-2xl border flex items-center justify-between ${
                  !st.sends_text_off_machine
                    ? 'bg-[#5ec2a8]/10 border-[#5ec2a8]/30 text-[#8ee3cb]'
                    : 'bg-[#e5b869]/10 border-[#e5b869]/30 text-[#f5d799]'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <ShieldCheck className="w-4 h-4 shrink-0" />
                  <div>
                    <div className="font-semibold">
                      {!st.sends_text_off_machine
                        ? 'Strict Local Privacy Mode Active'
                        : 'Cloud Hybrid Routing Active'}
                    </div>
                    <div className="text-[11px] opacity-80 font-mono">
                      Chain: {st.will_try.join(' → ') || 'none'}
                    </div>
                  </div>
                </div>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono uppercase bg-black/40 border border-white/10">
                  {st.mode}
                </span>
              </div>

              {/* Mode Selector */}
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => updateCfg.mutate({ mode: 'offline' })}
                  disabled={updateCfg.isPending}
                  className={`p-4 rounded-2xl border text-left transition-all cursor-pointer ${
                    st.mode === 'offline'
                      ? 'bg-[#5ec2a8]/15 border-[#5ec2a8]/50 text-[#f3efe6] shadow-[0_0_20px_rgba(94,194,168,0.15)]'
                      : 'bg-white/[0.03] border-white/10 text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.06]'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <Cpu className="w-4 h-4 text-[#5ec2a8]" />
                    {st.mode === 'offline' && (
                      <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded bg-[#5ec2a8]/20 text-[#5ec2a8]">
                        Active
                      </span>
                    )}
                  </div>
                  <div className="font-semibold text-sm text-[#f3efe6]">Offline (Ollama)</div>
                  <p className="text-[11px] text-zinc-400 mt-1">
                    Zero data leaves your machine. Uses local weights.
                  </p>
                </button>

                <button
                  onClick={() => updateCfg.mutate({ mode: 'cloud' })}
                  disabled={updateCfg.isPending}
                  className={`p-4 rounded-2xl border text-left transition-all cursor-pointer ${
                    st.mode === 'cloud'
                      ? 'bg-[#e5b869]/15 border-[#e5b869]/50 text-[#f3efe6] shadow-[0_0_20px_rgba(229,184,105,0.15)]'
                      : 'bg-white/[0.03] border-white/10 text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.06]'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <Cloud className="w-4 h-4 text-[#e5b869]" />
                    {st.mode === 'cloud' && (
                      <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded bg-[#e5b869]/20 text-[#e5b869]">
                        Active
                      </span>
                    )}
                  </div>
                  <div className="font-semibold text-sm text-[#f3efe6]">Cloud + Fallback</div>
                  <p className="text-[11px] text-zinc-400 mt-1">
                    Tries Groq / Gemini first, falls back to local Ollama.
                  </p>
                </button>
              </div>

              {/* Local Ollama Model Selector */}
              <div className="p-4 rounded-2xl bg-white/[0.03] border border-white/10 space-y-2.5">
                <div className="flex items-center justify-between font-mono">
                  <span className="text-zinc-300 font-semibold">Local Ollama Model</span>
                  <span className="text-[10px] text-[#e5b869]">
                    {st.ollama.installed.length} installed
                  </span>
                </div>
                {st.ollama.installed.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {st.ollama.installed.map((m) => {
                      const active = st.ollama.selected === m
                      return (
                        <button
                          key={m}
                          onClick={() => updateCfg.mutate({ ollama_model: m })}
                          className={`px-3 py-1.5 rounded-xl font-mono text-xs transition-all cursor-pointer ${
                            active
                              ? 'bg-[#e5b869]/20 text-[#e5b869] border border-[#e5b869]/50 font-semibold'
                              : 'bg-black/40 text-zinc-400 hover:text-white border border-white/10'
                          }`}
                        >
                          {m}
                        </button>
                      )
                    })}
                  </div>
                ) : (
                  <p className="text-zinc-500 font-mono text-[11px]">
                    {st.ollama.error || 'No local Ollama models detected.'}
                  </p>
                )}
              </div>

              {/* Cloud Providers Overview */}
              <div className="p-4 rounded-2xl bg-white/[0.03] border border-white/10 space-y-2">
                <div className="flex items-center justify-between font-mono">
                  <span className="text-zinc-300 font-semibold">Cloud Providers</span>
                  <button
                    onClick={() =>
                      updateCfg.mutate({
                        cloud_order: [...st.cloud.order].reverse(),
                      })
                    }
                    className="text-[10px] text-[#e07a5f] hover:underline cursor-pointer"
                  >
                    Swap Priority ({st.cloud.order.join(' → ')})
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-1">
                  {Object.entries(st.cloud.providers).map(([name, info]) => (
                    <div
                      key={name}
                      className="p-2.5 rounded-xl bg-black/30 border border-white/5 font-mono flex items-center justify-between"
                    >
                      <div>
                        <div className="uppercase text-[#f3efe6] font-semibold text-[11px]">{name}</div>
                        <div className="text-[10px] text-zinc-500 truncate max-w-[130px]">
                          {info.model}
                        </div>
                      </div>
                      <span
                        className={`text-[9px] px-1.5 py-0.5 rounded ${
                          info.configured
                            ? 'bg-[#5ec2a8]/15 text-[#5ec2a8]'
                            : 'bg-white/5 text-zinc-500'
                        }`}
                      >
                        {info.configured ? 'KEY SET' : 'NO KEY'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Live Ping / Round-Trip Test */}
              <div className="pt-1">
                <button
                  onClick={() => pingLLM.mutate()}
                  disabled={pingLLM.isPending}
                  className="w-full py-2.5 rounded-xl bg-gradient-to-r from-[#e5b869] to-[#c8963e] text-[#080a0a] font-semibold text-xs flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(229,184,105,0.28)] hover:opacity-95 disabled:opacity-50 transition-all cursor-pointer"
                >
                  {pingLLM.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Pinging Active LLM Router…</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      <span>Ping Neural Core (Test Round-Trip Latency)</span>
                    </>
                  )}
                </button>

                {testResult && (
                  <div
                    className={`mt-3 p-3 rounded-xl border font-mono text-[11px] flex items-start gap-2.5 ${
                      testResult.ok
                        ? 'bg-[#5ec2a8]/10 border-[#5ec2a8]/40 text-[#8ee3cb]'
                        : 'bg-[#d9534f]/15 border-[#d9534f]/40 text-[#f08a86]'
                    }`}
                  >
                    {testResult.ok ? (
                      <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-[#5ec2a8]" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-[#d9534f]" />
                    )}
                    <div className="min-w-0 flex-1">
                      {testResult.ok ? (
                        <>
                          <div className="font-semibold text-[#f3efe6]">
                            {testResult.provider?.toUpperCase()} ({testResult.model}) responded in{' '}
                            <span className="text-[#5ec2a8]">{testResult.latency_ms} ms</span>
                          </div>
                          <div className="text-zinc-300 mt-0.5">Reply: "{testResult.reply}"</div>
                        </>
                      ) : (
                        <div>{testResult.error}</div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
