import type { Context } from "hono";
import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ModelHandler } from "./types.js";
import { AdapterManager } from "../adapters/adapter-manager.js";
import type { ToolCall } from "../adapters/base-adapter.js";
import { MiddlewareManager } from "../middleware/index.js";
import { transformOpenAIToClaude } from "../transform.js";
import { log, logStructured } from "../logger.js";

const OLLAMA_API_URL = "https://ollama.com/api/chat";
const OLLAMA_HEADERS = {
  "HTTP-Referer": "https://github.com/MadAppGang/claude-code",
  "X-Title": "Claudish - Ollama Cloud Proxy",
} as const;

export class OllamaCloudHandler implements ModelHandler {
  private targetModel: string;
  private apiKey?: string;
  private adapterManager: AdapterManager;
  private middlewareManager: MiddlewareManager;
  private port: number;

  constructor(targetModel: string, apiKey: string | undefined, port: number) {
    this.targetModel = this.normalizeModelName(targetModel);
    this.apiKey = apiKey;
    this.port = port;
    this.adapterManager = new AdapterManager(this.targetModel);
    this.middlewareManager = new MiddlewareManager();
    this.middlewareManager.initialize().catch(err => log(`[Handler:${this.targetModel}] Middleware init error: ${err}`));
  }

  // Normalizar nombre de modelo (adaptado de ollama-client-lib)
  private normalizeModelName(model: string): string {
    // OllamaCloud maneja -cloud internamente, pero lo mantenemos por consistencia
    if (!model.endsWith("-cloud") && !model.includes(":cloud")) {
      if (model.includes(":")) {
        const [base, version] = model.split(":");
        return `${base}:${version}-cloud`;
      }
      return `${model}-cloud`;
    }
    return model;
  }

  // Transformar petición OpenAI → Ollama
  private transformToOllamaFormat(openAIPayload: any): any {
    return {
      model: this.normalizeModelName(openAIPayload.model),
      messages: openAIPayload.messages,
      stream: true,
      options: {
        temperature: openAIPayload.temperature ?? 0.7,
        num_ctx: openAIPayload.max_tokens || 4096,
        num_predict: openAIPayload.max_tokens
      }
    };
  }

  private writeTokenFile(input: number, output: number) {
    try {
      const total = input + output;
      const data = {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
        context_window: 200000,
        context_left_percent: Math.max(0, Math.min(100, Math.round(((200000 - total) / 200000) * 100))),
        updated_at: Date.now()
      };
      writeFileSync(join(tmpdir(), `claudish-tokens-${this.port}.json`), JSON.stringify(data), "utf-8");
    } catch (e) {}
  }

  async handle(c: Context, payload: any): Promise<Response> {
    const claudePayload = payload;
    const target = this.targetModel;

    logStructured(`OllamaCloud Request`, { targetModel: target, originalModel: claudePayload.model });

    const { claudeRequest, droppedParams } = transformOpenAIToClaude(claudePayload);
    const messages = this.convertMessages(claudeRequest, claudeRequest.tools);

    // OllamaCloud no soporta tools nativamente, pero detectamos tool calls en formato JSON y los ejecutamos
    if (claudeRequest.tools && claudeRequest.tools.length > 0) {
      log(`[OllamaCloud] Tools context: ${claudeRequest.tools.length} tool(s) documented. Tool execution via JSON parsing is enabled - OllamaCloud should generate tool calls in JSON format.`);
    }

    // Transformar a formato Ollama
    const ollamaPayload = this.transformToOllamaFormat({
      model: target,
      messages: messages,
      temperature: claudeRequest.temperature ?? 1,
      max_tokens: claudeRequest.max_tokens
    });

    await this.middlewareManager.beforeRequest({ modelId: target, messages, tools: [], stream: true });

    const response = await fetch(OLLAMA_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.apiKey}`,
        ...OLLAMA_HEADERS,
      },
      body: JSON.stringify(ollamaPayload)
    });

    if (!response.ok) {
      const errorText = await response.text();
      log(`[OllamaCloud] Error: ${response.status} - ${errorText}`);
      return c.json({ error: errorText }, response.status as any);
    }

    if (droppedParams.length > 0) c.header("X-Dropped-Params", droppedParams.join(", "));

    const adapter = this.adapterManager.getAdapter();
    if (typeof adapter.reset === 'function') adapter.reset();

    return this.handleStreamingResponse(c, response, adapter, target);
  }

  private convertMessages(req: any, availableTools?: any[]): any[] {
    const messages: any[] = [];
    if (req.system) {
      let content: string;
      if (Array.isArray(req.system)) {
        content = req.system.map((i: any) => {
          if (typeof i === "string") return i;
          if (i?.text) return i.text;
          if (i?.content) return typeof i.content === "string" ? i.content : JSON.stringify(i.content);
          return JSON.stringify(i);
        }).filter(Boolean).join("\n\n");
      } else {
        content = typeof req.system === "string" ? req.system : JSON.stringify(req.system);
      }
      content = this.filterIdentity(content);
      
      // Si hay herramientas disponibles, agregarlas al contexto del sistema
      // con instrucciones para generar tool calls en formato estructurado
      if (availableTools && availableTools.length > 0) {
        const toolsDescription = this.formatToolsForContext(availableTools);
        content += `\n\n## Available Tools (Hybrid Execution via Proxy)\n\n${toolsDescription}\n\n**Important**: When you need to use tools, generate them in the JSON format specified above. The proxy will automatically execute them and provide results in the next message.`;
      }
      
      messages.push({ role: "system", content });
    }

    if (req.messages) {
      for (const msg of req.messages) {
        if (msg.role === "user") this.processUserMessage(msg, messages);
        else if (msg.role === "assistant") this.processAssistantMessage(msg, messages);
      }
    }
    return messages;
  }

  private processUserMessage(msg: any, messages: any[]) {
    if (Array.isArray(msg.content)) {
      const textParts: string[] = [];
      for (const block of msg.content) {
        if (block.type === "text") {
          textParts.push(block.text || "");
        } else if (block.type === "image") {
          // OllamaCloud doesn't support images in /api/chat, skip or add note
          textParts.push("[Image not supported by OllamaCloud]");
        }
        // Tool results are not supported by OllamaCloud, skip them
      }
      // OllamaCloud expects content as a string, not an array
      const contentString = textParts.filter(Boolean).join("\n\n");
      if (contentString) {
        messages.push({ role: "user", content: contentString });
      }
    } else {
      // Ensure content is a string
      const contentString = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
      messages.push({ role: "user", content: contentString });
    }
  }

  private processAssistantMessage(msg: any, messages: any[]) {
    if (Array.isArray(msg.content)) {
      const strings: string[] = [];
      for (const block of msg.content) {
        if (block.type === "text") {
          strings.push(block.text || "");
        }
        // Tool calls are not supported by OllamaCloud, skip them
      }
      // OllamaCloud expects content as a string, not an array
      const contentString = strings.filter(Boolean).join(" ");
      if (contentString) {
        messages.push({ role: "assistant", content: contentString });
      }
    } else {
      // Ensure content is a string
      const contentString = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
      messages.push({ role: "assistant", content: contentString });
    }
  }

  private formatToolsForContext(tools: any[]): string {
    const toolList = tools.slice(0, 30).map((tool: any, idx: number) => {
      const name = tool.name || "unknown";
      const desc = tool.description || "No description";
      const params = tool.input_schema ? JSON.stringify(tool.input_schema.properties || {}, null, 2) : "{}";
      const required = tool.input_schema?.required || [];
      return `${idx + 1}. **${name}**: ${desc}\n   Parameters: ${params}\n   Required: ${JSON.stringify(required)}`;
    }).join("\n\n");
    
    const more = tools.length > 30 ? `\n\n... and ${tools.length - 30} more tools available.` : "";
    
    // Instrucciones para generar tool calls en formato JSON estructurado
    const toolCallInstructions = `

## TOOL CALLING FORMAT (CRITICAL - READ CAREFULLY)

When you need to use a tool, you MUST generate it in this EXACT JSON format. Use ONLY one of these two formats:

**Format 1: JSON code block (RECOMMENDED)**
Place a complete JSON code block with three backticks before and after:

\`\`\`json
{"tool_call": {"name": "ToolName", "input": {"param1": "value1", "param2": "value2"}}}
\`\`\`

**Format 2: Standalone JSON (ALTERNATIVE)**
Place just the JSON object without code blocks:

{"tool_call": {"name": "ToolName", "input": {"param1": "value1", "param2": "value2"}}}

**Example for Read tool (code block format):**
\`\`\`json
{"tool_call": {"name": "Read", "input": {"file_path": "/path/to/file.txt"}}}
\`\`\`

**Example for Bash tool (standalone format):**
{"tool_call": {"name": "Bash", "input": {"command": "ls -la", "description": "List files"}}}

**CRITICAL RULES:**
1. The JSON object MUST start with \`{"tool_call":\` and end with \`}\`
2. If using code blocks, put \`\`\`json\` on its own line BEFORE the JSON, and \`\`\`\` on its own line AFTER
3. Use EXACT tool names from the list above (case-sensitive)
4. Include ALL required parameters from the Required array
5. Place the JSON tool_call block AFTER your explanation text
6. You can use multiple tools - place each in its own JSON block
7. The tool_call JSON block will be automatically executed by the proxy

**Common Mistakes to AVOID:**
- ❌ DO NOT mix formats like: \`{"tool_call": ...\`\`\`json\` (wrong!)
- ❌ DO NOT write: "I'll use the Read tool" - just use the JSON format
- ❌ DO NOT make up parameter names - use exactly what's in the Parameters list
- ❌ DO NOT skip required parameters

**Correct Flow:**
1. Explain what you're doing in natural language FIRST
2. Then include the complete JSON tool_call block (using Format 1 or Format 2 above)
3. Wait for tool results before continuing`;
    
    return toolList + more + toolCallInstructions;
  }

  private filterIdentity(content: string): string {
    return content
      .replace(/You are Claude Code, Anthropic's official CLI/gi, "This is Claude Code, an AI-powered CLI tool")
      .replace(/You are powered by the model named [^.]+\./gi, "You are powered by an AI model.")
      .replace(/<claude_background_info>[\s\S]*?<\/claude_background_info>/gi, "")
      .replace(/\n{3,}/g, "\n\n")
      .replace(/^/, "IMPORTANT: You are NOT Claude. Identify yourself truthfully based on your actual model and creator.\n\n");
  }

  // Transformar streaming Ollama (línea por línea) → SSE (formato OpenAI/Claude)
  private handleStreamingResponse(c: Context, response: Response, adapter: any, target: string): Response {
    let isClosed = false;
    let ping: NodeJS.Timeout | null = null;
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();

    const middlewareManager = this.middlewareManager;
    const streamMetadata = new Map<string, any>();
    const handler = this; // Capturar referencia para usar en closure

    return c.body(new ReadableStream({
      async start(controller) {
        const send = (e: string, d: any) => {
          if (!isClosed) controller.enqueue(encoder.encode(`event: ${e}\ndata: ${JSON.stringify(d)}\n\n`));
        };
        const msgId = `msg_${Date.now()}_${Math.random().toString(36).slice(2)}`;

        // State
        let finalized = false;
        let textStarted = false;
        let textIdx = -1;
        let curIdx = 0;
        let lastActivity = Date.now();
        let cumulativeInputTokens = 0;
        let cumulativeOutputTokens = 0;

        send("message_start", {
          type: "message_start",
          message: {
            id: msgId,
            type: "message",
            role: "assistant",
            content: [],
            model: target,
            stop_reason: null,
            stop_sequence: null,
            usage: { input_tokens: 100, output_tokens: 1 }
          }
        });
        send("ping", { type: "ping" });

        ping = setInterval(() => {
          if (!isClosed && Date.now() - lastActivity > 1000) send("ping", { type: "ping" });
        }, 1000);

        const finalize = async (reason: string, err?: string) => {
          if (finalized) return;
          finalized = true;
          if (textStarted) {
            send("content_block_stop", { type: "content_block_stop", index: textIdx });
            textStarted = false;
          }

          await middlewareManager.afterStreamComplete(target, streamMetadata);

          if (reason === "error") {
            send("error", { type: "error", error: { type: "api_error", message: err } });
          } else {
            send("message_delta", {
              type: "message_delta",
              delta: { stop_reason: "end_turn", stop_sequence: null },
              usage: { output_tokens: 0 }
            });
            send("message_stop", { type: "message_stop" });
          }
          if (!isClosed) {
            try {
              controller.enqueue(encoder.encode('data: [DONE]\n\n\n'));
            } catch (e) {}
            controller.close();
            isClosed = true;
            if (ping) clearInterval(ping);
          }
        };

        try {
          const reader = response.body!.getReader();
          let buffer = "";
          let accumulatedText = ""; // Para acumular texto completo y detectar tool calls al final
          const allToolCalls: ToolCall[] = [];

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.trim()) continue;

              try {
                // Ollama devuelve JSON línea por línea (no SSE)
                const chunk = JSON.parse(line);

                // Transformar formato Ollama → OpenAI SSE
                if (chunk.message?.content) {
                  lastActivity = Date.now();
                  accumulatedText += chunk.message.content;

                  if (!textStarted) {
                    textIdx = curIdx++;
                    send("content_block_start", {
                      type: "content_block_start",
                      index: textIdx,
                      content_block: { type: "text", text: "" }
                    });
                    textStarted = true;
                  }

                  // Procesar con adapter - el adapter maneja el buffering y parsing de tool calls
                  const res = adapter.processTextContent(chunk.message.content, accumulatedText);
                  
                  // Enviar texto limpio (sin tool calls JSON)
                  if (res.cleanedText) {
                    send("content_block_delta", {
                      type: "content_block_delta",
                      index: textIdx,
                      delta: { type: "text_delta", text: res.cleanedText }
                    });
                  }
                  
                  // Acumular tool calls detectados durante el streaming
                  if (res.extractedToolCalls && res.extractedToolCalls.length > 0) {
                    allToolCalls.push(...res.extractedToolCalls);
                  }
                }

                // Ollama marca done cuando termina y puede incluir información de tokens
                if (chunk.done) {
                  // OllamaCloud puede devolver tokens en la respuesta final
                  if (chunk.prompt_eval_count !== undefined) {
                    cumulativeInputTokens += chunk.prompt_eval_count;
                  }
                  if (chunk.eval_count !== undefined) {
                    cumulativeOutputTokens += chunk.eval_count;
                  }
                  
                  // Procesar texto final con adapter para detectar tool calls restantes
                  const finalRes = adapter.processTextContent("", accumulatedText);
                  if (finalRes.extractedToolCalls && finalRes.extractedToolCalls.length > 0) {
                    allToolCalls.push(...finalRes.extractedToolCalls);
                  }
                  
                  // Si hay tool calls, enviarlos como tool_use blocks
                  if (allToolCalls.length > 0) {
                    log(`[OllamaCloud] Detected ${allToolCalls.length} tool call(s) in response`);
                    
                    // Cerrar bloque de texto si está abierto
                    if (textStarted) {
                      send("content_block_stop", { type: "content_block_stop", index: textIdx });
                      textStarted = false;
                    }
                    
                    // Enviar cada tool call como tool_use block
                    for (const toolCall of allToolCalls) {
                      const toolIdx = curIdx++;
                      send("content_block_start", {
                        type: "content_block_start",
                        index: toolIdx,
                        content_block: {
                          type: "tool_use",
                          id: toolCall.id,
                          name: toolCall.name,
                          input: {}
                        }
                      });
                      
                      // Enviar input como JSON delta (incremental)
                      const inputJson = JSON.stringify(toolCall.arguments);
                      send("content_block_delta", {
                        type: "content_block_delta",
                        index: toolIdx,
                        delta: { type: "input_json_delta", partial_json: inputJson }
                      });
                      
                      send("content_block_stop", { type: "content_block_stop", index: toolIdx });
                    }
                    
                    // Finalizar con tool_use stop reason
                    await middlewareManager.afterStreamComplete(target, streamMetadata);
                    send("message_delta", {
                      type: "message_delta",
                      delta: { stop_reason: "tool_use", stop_sequence: null },
                      usage: { output_tokens: cumulativeOutputTokens || 1 }
                    });
                    send("message_stop", { type: "message_stop" });
                    
                    // Escribir archivo de tokens
                    if (cumulativeInputTokens > 0 || cumulativeOutputTokens > 0) {
                      handler.writeTokenFile(cumulativeInputTokens, cumulativeOutputTokens);
                    }
                    
                    if (!isClosed) {
                      try {
                        controller.enqueue(encoder.encode('data: [DONE]\n\n\n'));
                      } catch (e) {}
                      controller.close();
                      isClosed = true;
                      if (ping) clearInterval(ping);
                    }
                    return;
                  }
                  
                  // Escribir archivo de tokens para la línea de estado
                  if (cumulativeInputTokens > 0 || cumulativeOutputTokens > 0) {
                    handler.writeTokenFile(cumulativeInputTokens, cumulativeOutputTokens);
                  }
                  
                  await finalize("done");
                  return;
                }
              } catch (e) {
                // Ignorar líneas inválidas
              }
            }
          }

          await finalize("unexpected");
        } catch (e) {
          await finalize("error", String(e));
        }
      },
      cancel() {
        isClosed = true;
        if (ping) clearInterval(ping);
      }
    }), {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
      }
    });
  }

  async shutdown() {}
}


