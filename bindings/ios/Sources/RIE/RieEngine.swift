import CRie
import Foundation

/// Swift wrapper around the Rust Inference Engine.
///
/// ```swift
/// let engine = try RieEngine(modelPath: Bundle.main.path(forResource: "model", ofType: "gguf")!)
/// let stats = try engine.chat([.init(role: .user, content: "Hello!")]) { piece in
///     DispatchQueue.main.async { label.text! += piece }
///     return true
/// }
/// ```
/// Calls block: run them off the main thread. `cancel()` is safe from any thread.
public final class RieEngine {
    public enum Role: String { case system, user, assistant }
    public struct Message { public var role: Role; public var content: String
        public init(role: Role, content: String) { self.role = role; self.content = content } }

    public struct Sampling {
        public var maxTokens = 512, temperature: Float = 0.7, topK = 40
        public var topP: Float = 0.95, minP: Float = 0.05, repeatPenalty: Float = 1.1, seed: UInt64 = 0
        public init() {}
    }

    public struct Stats {
        public let promptTokens, reusedTokens, generatedTokens: Int
        public let promptTokensPerSecond, generatedTokensPerSecond: Double
        public let stopReason: Int32
    }

    public struct EngineError: Error, CustomStringConvertible { public let description: String }

    private let handle: OpaquePointer

    /// - Parameters:
    ///   - threads: 0 = auto (P-cores)
    ///   - useGPU: Metal backend (library built with the `gpu` feature)
    public init(modelPath: String, threads: Int = 0, contextSize: Int = 4096, useGPU: Bool = false) throws {
        var cfg = rie_default_config()
        cfg.n_threads = Int32(threads)
        cfg.n_ctx = Int32(contextSize)
        cfg.use_gpu = useGPU ? 1 : 0
        var err = [CChar](repeating: 0, count: 512)
        guard let h = rie_load(modelPath, &cfg, &err, err.count) else {
            throw EngineError(description: String(cString: err))
        }
        handle = h
    }

    deinit { rie_free(handle) }

    public static var deviceInfo: String {
        let p = rie_device_info()!; defer { rie_string_free(p) }; return String(cString: p)
    }

    public var modelInfo: String {
        let p = rie_model_info(handle)!; defer { rie_string_free(p) }; return String(cString: p)
    }

    public func cancel() { rie_cancel(handle) }
    public func reset() { rie_reset(handle) }

    private final class Box { let f: (String) -> Bool; init(_ f: @escaping (String) -> Bool) { self.f = f } }

    private static let trampoline: RieTokenCallback = { piece, user in
        let box = Unmanaged<Box>.fromOpaque(user!).takeUnretainedValue()
        return box.f(String(cString: piece!)) ? 1 : 0
    }

    private func cSampling(_ s: Sampling) -> RieSampling {
        var c = rie_default_sampling()
        c.max_tokens = Int32(s.maxTokens); c.temperature = s.temperature; c.top_k = Int32(s.topK)
        c.top_p = s.topP; c.min_p = s.minP; c.repeat_penalty = s.repeatPenalty; c.seed = s.seed
        return c
    }

    private func stats(_ s: RieStats) -> Stats {
        Stats(promptTokens: Int(s.prompt_tokens), reusedTokens: Int(s.reused_tokens),
              generatedTokens: Int(s.generated_tokens), promptTokensPerSecond: s.prompt_tps,
              generatedTokensPerSecond: s.gen_tps, stopReason: s.stop_reason)
    }

    @discardableResult
    public func generate(_ prompt: String, sampling: Sampling = .init(), onToken: @escaping (String) -> Bool) throws -> Stats {
        let box = Unmanaged.passRetained(Box(onToken)); defer { box.release() }
        var cs = cSampling(sampling), st = RieStats(), err = [CChar](repeating: 0, count: 512)
        let rc = rie_generate(handle, prompt, &cs, RieEngine.trampoline, box.toOpaque(), &st, &err, err.count)
        if rc != 0 { throw EngineError(description: String(cString: err)) }
        return stats(st)
    }

    @discardableResult
    public func chat(_ messages: [Message], sampling: Sampling = .init(), onToken: @escaping (String) -> Bool) throws -> Stats {
        let box = Unmanaged.passRetained(Box(onToken)); defer { box.release() }
        var cs = cSampling(sampling), st = RieStats(), err = [CChar](repeating: 0, count: 512)
        let roles = messages.map { strdup($0.role.rawValue) }
        let contents = messages.map { strdup($0.content) }
        defer { roles.forEach { free($0) }; contents.forEach { free($0) } }
        let r = roles.map { UnsafePointer<CChar>($0) }, c = contents.map { UnsafePointer<CChar>($0) }
        let rc = r.withUnsafeBufferPointer { rp in c.withUnsafeBufferPointer { cp in
            rie_chat(handle, rp.baseAddress, cp.baseAddress, Int32(messages.count), &cs,
                     RieEngine.trampoline, box.toOpaque(), &st, &err, err.count)
        } }
        if rc != 0 { throw EngineError(description: String(cString: err)) }
        return stats(st)
    }

    public func bench(promptTokens: Int = 128, genTokens: Int = 64) -> (pp: Double, tg: Double)? {
        var pp = 0.0, tg = 0.0
        return rie_bench(handle, Int32(promptTokens), Int32(genTokens), &pp, &tg) == 0 ? (pp, tg) : nil
    }
}
