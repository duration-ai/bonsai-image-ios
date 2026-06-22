import Foundation
import Tokenizers

// Tokenization for the on-device encoder: loads the HF tokenizer directory
// (tokenizer.json + tokenizer_config.json — same files the Python pipeline
// uses), renders the parity-pinned chat template around the user prompt, and
// returns ids for Qwen3Encoder.encode. The template rendering is hard-coded
// (prefix/suffix from the fixture meta) rather than re-implementing Jinja.
public enum QwenTokenizer {
    public static func load(dir: URL) throws -> Tokenizer {
        // AutoTokenizer.from(modelFolder:) is async (Hub-shaped API) but purely
        // local-file here; bridge to sync — callers are on background queues.
        // Box is @unchecked Sendable: the semaphore orders the write before the read.
        final class Box: @unchecked Sendable { var result: Result<Tokenizer, Error>? }
        let box = Box()
        let sem = DispatchSemaphore(value: 0)
        Task.detached {
            do { box.result = .success(try await AutoTokenizer.from(modelFolder: dir)) }
            catch { box.result = .failure(error) }
            sem.signal()
        }
        sem.wait()
        return try box.result!.get()
    }

    /// prompt -> rendered-template token ids (verified == HF apply_chat_template
    /// + encode in the qwen parity mode).
    public static func promptIds(_ tokenizer: Tokenizer, prompt: String) -> [Int32] {
        let text = Qwen3Encoder.templatePrefix + prompt + Qwen3Encoder.templateSuffix
        return tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }
}
