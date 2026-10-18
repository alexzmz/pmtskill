// Dependency-free coverage of the actual TypeScript provider when Bun is not
// installed: node --experimental-transform-types --test tests/test_skvm_provider_protocol.mjs
import { afterEach, test } from "node:test"
import assert from "node:assert/strict"
import { OpenAICompatibleProvider } from "../libs/skvm/src/providers/openai-compatible.ts"

const realFetch = globalThis.fetch
afterEach(() => { globalThis.fetch = realFetch })

function response(reason, calls = true, args = '{"path":"input.txt"}') {
  return {
    choices: [{ finish_reason: reason, message: {
      role: "assistant", content: "<tool_call>read_file</tool_call>",
      ...(calls ? { tool_calls: [{ id: "read-1", type: "function", function: { name: "read_file", arguments: args } }] } : {}),
    } }], usage: { prompt_tokens: 652, completion_tokens: 73 },
  }
}

const provider = () => new OpenAICompatibleProvider({ apiKey: "fake", model: "adapter_control01", baseUrl: "http://local/v1" })
const params = { messages: [{ role: "user", content: "read input.txt" }] }
function serve(body) {
  globalThis.fetch = async () => new Response(JSON.stringify(body), { status: 200 })
}

for (const reason of ["stop", null, undefined, "tool_calls"]) {
  test(`valid parsed tool with finish_reason=${reason} remains executable`, async () => {
    serve(response(reason))
    const result = await provider().complete(params)
    assert.equal(result.stopReason, "tool_use")
    assert.deepEqual(result.toolCalls[0], { id: "read-1", name: "read_file", arguments: { path: "input.txt" } })
  })
}

test("textual markup without structured tool_calls is not executable", async () => {
  serve(response("stop", false))
  const result = await provider().complete(params)
  assert.equal(result.stopReason, "end_turn")
  assert.deepEqual(result.toolCalls, [])
})

for (const [reason, expected] of [["length", "max_tokens"], ["content_filter", "end_turn"]]) {
  test(`${reason} is not reclassified as tool_use`, async () => {
    serve(response(reason))
    assert.equal((await provider().complete(params)).stopReason, expected)
  })
}

test("malformed JSON tool arguments still raise", async () => {
  serve(response("stop", true, "{"))
  await assert.rejects(provider().complete(params), { name: "ToolArgumentsParseError" })
})

test("tool result is returned to the same adapter for the next turn", async () => {
  const bodies = []
  globalThis.fetch = async (_url, options) => {
    bodies.push(JSON.parse(options.body))
    return new Response(JSON.stringify(response("stop", bodies.length === 1)), { status: 200 })
  }
  const model = provider()
  const first = await model.complete(params)
  assert.equal(first.stopReason, "tool_use")
  const second = await model.completeWithToolResults(params, [{ toolCallId: "read-1", content: "a:80" }], first)
  assert.equal(second.stopReason, "end_turn")
  assert.equal(bodies[1].model, "adapter_control01")
  assert.equal(bodies[1].messages.at(-1).role, "tool")
  assert.equal(bodies[1].messages.at(-1).tool_call_id, "read-1")
  assert.equal(bodies[1].messages.at(-1).content, "a:80")
})
