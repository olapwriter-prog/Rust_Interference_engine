package ai.rie

import java.io.Closeable

/**
 * Kotlin wrapper around librie.so.
 *
 * ```kotlin
 * val engine = RieEngine.load(File(filesDir, "qwen2.5-1.5b-instruct-q4_0.gguf").path)
 * val stats = engine.chat(listOf(ChatMessage("user", "Hello!"))) { piece ->
 *     runOnUiThread { textView.append(piece) }
 *     true   // return false to stop
 * }
 * ```
 *
 * All calls block; run them on a background thread (e.g. `Dispatchers.Default`).
 * [cancel] may be called from any thread.
 */
class RieEngine private constructor(private var handle: Long) : Closeable {

    fun interface TokenCallback {
        /** Called with each piece of generated text. Return false to stop. */
        fun onToken(piece: String): Boolean
    }

    data class ChatMessage(val role: String, val content: String)

    data class Sampling(
        val maxTokens: Int = 512,
        val temperature: Float = 0.7f,
        val topK: Int = 40,
        val topP: Float = 0.95f,
        val minP: Float = 0.05f,
        val repeatPenalty: Float = 1.1f,
        val seed: Long = 0,
    )

    /** JSON: prompt_tokens, reused_tokens, generated_tokens, prompt_tps, gen_tps, stop_reason */
    fun generate(prompt: String, sampling: Sampling = Sampling(), callback: TokenCallback): String =
        nativeGenerate(checkHandle(), prompt, null, null, sampling.maxTokens, sampling.temperature,
            sampling.topK, sampling.topP, sampling.minP, sampling.repeatPenalty, sampling.seed, callback)

    /** Chat completion; the shared prefix with the previous call is served from the KV cache. */
    fun chat(messages: List<ChatMessage>, sampling: Sampling = Sampling(), callback: TokenCallback): String =
        nativeGenerate(checkHandle(), null, messages.map { it.role }.toTypedArray(),
            messages.map { it.content }.toTypedArray(), sampling.maxTokens, sampling.temperature,
            sampling.topK, sampling.topP, sampling.minP, sampling.repeatPenalty, sampling.seed, callback)

    fun cancel() { if (handle != 0L) nativeCancel(handle) }
    fun reset() = nativeReset(checkHandle())
    fun bench(promptTokens: Int = 128, genTokens: Int = 64): String = nativeBench(checkHandle(), promptTokens, genTokens)
    val modelInfo: String get() = nativeModelInfo(checkHandle())

    override fun close() {
        if (handle != 0L) { nativeFree(handle); handle = 0L }
    }

    private fun checkHandle(): Long { check(handle != 0L) { "engine closed" }; return handle }

    companion object {
        init { System.loadLibrary("rie") }

        /**
         * @param threads 0 = auto (performance cores only, pinned)
         * @param useGpu run on the Vulkan GPU backend (library built with `--features gpu`)
         * @param kvQ8 8-bit KV cache: ~47% less memory for long conversations
         */
        @JvmStatic
        fun load(path: String, threads: Int = 0, contextSize: Int = 4096, useGpu: Boolean = false, kvQ8: Boolean = false): RieEngine =
            RieEngine(nativeLoad(path, threads, contextSize, useGpu, kvQ8))

        @JvmStatic val deviceInfo: String get() = nativeDeviceInfo()

        @JvmStatic private external fun nativeLoad(path: String, threads: Int, ctx: Int, gpu: Boolean, kvQ8: Boolean): Long
        @JvmStatic private external fun nativeFree(h: Long)
        @JvmStatic private external fun nativeCancel(h: Long)
        @JvmStatic private external fun nativeReset(h: Long)
        @JvmStatic private external fun nativeGenerate(
            h: Long, prompt: String?, roles: Array<String>?, contents: Array<String>?,
            maxTokens: Int, temperature: Float, topK: Int, topP: Float, minP: Float,
            repeatPenalty: Float, seed: Long, callback: TokenCallback?,
        ): String
        @JvmStatic private external fun nativeBench(h: Long, pp: Int, tg: Int): String
        @JvmStatic private external fun nativeDeviceInfo(): String
        @JvmStatic private external fun nativeModelInfo(h: Long): String
    }
}
