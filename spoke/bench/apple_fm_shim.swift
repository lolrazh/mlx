// CLI shim for benchmarking Apple's on-device Foundation Model.
// Reads JSON lines {"system": ..., "input": ...} on stdin; writes JSON lines
// {"output": ..., "seconds": ...} on stdout. One fresh session per item so
// no context leaks between test cases.
//
// Build: swiftc -parse-as-library -O apple_fm_shim.swift -o apple_fm_shim
// Requires macOS 26+ with Apple Intelligence enabled.

import Foundation
import FoundationModels

struct Item: Codable {
    let system: String
    let input: String
}

struct Out: Codable {
    let output: String
    let seconds: Double
}

@main
struct Shim {
    static func main() async {
        // Permissive guardrails: Spoke transforms user-dictated content
        // verbatim (incl. profanity); default guardrails refuse innocuous
        // transcript-cleanup requests.
        let model = SystemLanguageModel(guardrails: .permissiveContentTransformations)
        guard case .available = model.availability else {
            FileHandle.standardError.write(
                "MODEL UNAVAILABLE: \(model.availability)\n".data(using: .utf8)!)
            exit(2)
        }

        var lines: [String] = []
        while let line = readLine(strippingNewline: true) {
            if !line.trimmingCharacters(in: .whitespaces).isEmpty {
                lines.append(line)
            }
        }

        let decoder = JSONDecoder()
        let encoder = JSONEncoder()

        for line in lines {
            guard let data = line.data(using: .utf8),
                  let item = try? decoder.decode(Item.self, from: data)
            else {
                FileHandle.standardError.write("SKIP unparseable line\n".data(using: .utf8)!)
                continue
            }
            let session = LanguageModelSession(model: model, instructions: item.system)
            let start = Date()
            var result: Out
            do {
                let options = GenerationOptions(temperature: 0.0)
                let resp = try await session.respond(to: item.input, options: options)
                result = Out(output: resp.content, seconds: Date().timeIntervalSince(start))
            } catch {
                result = Out(output: "<<ERROR: \(error)>>", seconds: Date().timeIntervalSince(start))
            }
            if let encoded = try? encoder.encode(result),
               let s = String(data: encoded, encoding: .utf8) {
                print(s)
                fflush(stdout)
            }
        }
    }
}
