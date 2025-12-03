/**
 * Ollama adapter for translating JSON tool calls to Claude Code tool_calls
 *
 * OllamaCloud models output tool calls in JSON format:
 * ```json
 * {"tool_call": {"name": "Read", "input": {"file_path": "/path/to/file.txt"}}}
 * ```
 *
 * This adapter translates that to Claude Code's expected tool_calls format.
 */

import { BaseModelAdapter, AdapterResult, ToolCall } from "./base-adapter";
import { log } from "../logger";

export class OllamaAdapter extends BaseModelAdapter {
  private textBuffer: string = "";
  private sentTextLength: number = 0; // Track how much text we've already sent

  processTextContent(
    textContent: string,
    _accumulatedText: string
  ): AdapterResult {
    // Accumulate text to handle JSON split across multiple chunks
    this.textBuffer += textContent;

    // Pattern 1: JSON code blocks with tool_call (properly formatted)
    const jsonBlockPattern = /```json\s*\n\s*\{\s*"tool_call"\s*:\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}\s*\}\s*```/gs;
    
    // Pattern 2: Malformed JSON mixed with markdown (e.g., {"tool_call```json)
    const malformedJsonPattern = /\{\s*"tool_call"\s*:\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}\s*\}\s*```/gs;
    
    // Pattern 3: Standalone JSON (without code blocks) - most flexible
    const standaloneJsonPattern = /\{\s*"tool_call"\s*:\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}\s*\}/gs;

    // Try all patterns
    const allMatches: Array<{ match: string; json: string }> = [];

    // Find properly formatted JSON code blocks
    let match: RegExpExecArray | null;
    while ((match = jsonBlockPattern.exec(this.textBuffer)) !== null) {
      allMatches.push({
        match: match[0],
        json: match[0].replace(/```json\s*|\s*```/g, '').trim()
      });
    }

    // Find malformed JSON mixed with markdown (e.g., {"tool_call": {...}}```json)
    while ((match = malformedJsonPattern.exec(this.textBuffer)) !== null) {
      const jsonPart = match[0].replace(/```.*$/, '').trim();
      allMatches.push({
        match: match[0],
        json: jsonPart
      });
    }

    // Find standalone JSON (only if not already in a code block)
    const codeBlockRanges: Array<{ start: number; end: number }> = [];
    let codeBlockMatch: RegExpExecArray | null;
    const codeBlockRegex = /```[^`]+```/g;
    while ((codeBlockMatch = codeBlockRegex.exec(this.textBuffer)) !== null) {
      codeBlockRanges.push({
        start: codeBlockMatch.index,
        end: codeBlockMatch.index + codeBlockMatch[0].length
      });
    }

    let standaloneMatch: RegExpExecArray | null;
    while ((standaloneMatch = standaloneJsonPattern.exec(this.textBuffer)) !== null) {
      // Check if this match is inside a code block
      const isInCodeBlock = codeBlockRanges.some(
        range => standaloneMatch!.index >= range.start && standaloneMatch!.index < range.end
      );

      if (!isInCodeBlock) {
        // Check if we already found this in a code block
        const alreadyFound = allMatches.some(m => m.match.includes(standaloneMatch![0]));
        if (!alreadyFound) {
          allMatches.push({ match: standaloneMatch[0], json: standaloneMatch[0] });
        }
      }
    }

    // If no matches found, check if we have a partial tool call
    if (allMatches.length === 0) {
      // Check if we have a partial tool call JSON
      const partialPattern = /\{\s*"tool_call"/;
      const hasPartialToolCall = partialPattern.test(this.textBuffer);

      if (hasPartialToolCall) {
        // Find where the partial tool call starts
        const partialMatch = this.textBuffer.match(/(.*?)\{\s*"tool_call"/);
        if (partialMatch) {
          // Send text before the partial tool call
          const textBeforePartial = partialMatch[1];
          const newText = textBeforePartial.slice(this.sentTextLength);
          this.sentTextLength = textBeforePartial.length;
          
          // Keep the partial tool call in buffer for next chunk
          this.textBuffer = this.textBuffer.slice(textBeforePartial.length);
          this.sentTextLength = 0; // Reset since we're starting a new buffer
          
          return {
            cleanedText: newText,
            extractedToolCalls: [],
            wasTransformed: false,
          };
        }
        
        // If we can't find the start, keep accumulating
        return {
          cleanedText: "",
          extractedToolCalls: [],
          wasTransformed: false,
        };
      }

      // Normal text, no tool calls - return only new text
      const newText = this.textBuffer.slice(this.sentTextLength);
      this.sentTextLength = this.textBuffer.length;
      
      return {
        cleanedText: newText,
        extractedToolCalls: [],
        wasTransformed: false,
      };
    }

    // Extract tool calls from JSON - try to fix common issues first
    const toolCalls: ToolCall[] = [];
    const seenToolCalls = new Set<string>(); // Avoid duplicates
    
    for (const { match: matchText, json: jsonText } of allMatches) {
      try {
        // Try to clean up common formatting issues
        let cleanedJson = jsonText.trim();
        
        // Remove any trailing markdown code block markers
        cleanedJson = cleanedJson.replace(/```[a-z]*\s*$/g, '').trim();
        
        // Try to extract JSON if it's wrapped in extra characters
        const jsonMatch = cleanedJson.match(/\{\s*"tool_call"\s*:[\s\S]*\}/);
        if (jsonMatch) {
          cleanedJson = jsonMatch[0];
        }
        
        const parsed = JSON.parse(cleanedJson);
        if (parsed.tool_call && parsed.tool_call.name && parsed.tool_call.input) {
          const toolKey = `${parsed.tool_call.name}:${JSON.stringify(parsed.tool_call.input)}`;
          if (seenToolCalls.has(toolKey)) {
            continue; // Skip duplicates
          }
          seenToolCalls.add(toolKey);
          
          const toolCall: ToolCall = {
            id: `ollama_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`,
            name: parsed.tool_call.name,
            arguments: parsed.tool_call.input,
          };
          toolCalls.push(toolCall);
          log(`[OllamaAdapter] Detected tool call: ${toolCall.name} with params: ${JSON.stringify(toolCall.arguments)}`);
        }
      } catch (e) {
        // Try to extract valid JSON from the malformed string
        try {
          // Look for JSON object boundaries
          const jsonStart = jsonText.indexOf('{"tool_call"');
          if (jsonStart >= 0) {
            let braceCount = 0;
            let jsonEnd = jsonStart;
            let foundStart = false;
            
            for (let i = jsonStart; i < jsonText.length; i++) {
              if (jsonText[i] === '{') {
                braceCount++;
                foundStart = true;
              } else if (jsonText[i] === '}') {
                braceCount--;
                if (foundStart && braceCount === 0) {
                  jsonEnd = i + 1;
                  break;
                }
              }
            }
            
            if (jsonEnd > jsonStart) {
              const extractedJson = jsonText.slice(jsonStart, jsonEnd);
              const parsed = JSON.parse(extractedJson);
              if (parsed.tool_call && parsed.tool_call.name && parsed.tool_call.input) {
                const toolKey = `${parsed.tool_call.name}:${JSON.stringify(parsed.tool_call.input)}`;
                if (!seenToolCalls.has(toolKey)) {
                  seenToolCalls.add(toolKey);
                  const toolCall: ToolCall = {
                    id: `ollama_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`,
                    name: parsed.tool_call.name,
                    arguments: parsed.tool_call.input,
                  };
                  toolCalls.push(toolCall);
                  log(`[OllamaAdapter] Recovered tool call from malformed JSON: ${toolCall.name}`);
                }
              }
            }
          }
        } catch (recoveryError) {
          // Failed to recover, log and skip
          log(`[OllamaAdapter] Failed to parse tool call JSON: ${jsonText.substring(0, 100)}... Error: ${e}`);
        }
      }
    }

    // Remove tool call JSON blocks from text
    let cleanedBuffer = this.textBuffer;
    for (const { match: matchText } of allMatches) {
      cleanedBuffer = cleanedBuffer.replace(matchText, '');
    }

    // Clean up extra whitespace
    cleanedBuffer = cleanedBuffer.replace(/\n{3,}/g, '\n\n');

    // Return only new cleaned text (incremental)
    const newCleanedText = cleanedBuffer.slice(this.sentTextLength);
    this.sentTextLength = cleanedBuffer.length;

    return {
      cleanedText: newCleanedText,
      extractedToolCalls: toolCalls,
      wasTransformed: toolCalls.length > 0,
    };
  }

  shouldHandle(modelId: string): boolean {
    // OllamaCloud models typically have -cloud suffix or :cloud tag
    return modelId.includes("-cloud") || modelId.includes(":cloud") || 
           (modelId.includes(":") && !modelId.includes("/"));
  }

  getName(): string {
    return "OllamaAdapter";
  }

  /**
   * Reset internal state (useful between requests)
   */
  reset(): void {
    this.textBuffer = "";
    this.sentTextLength = 0;
  }
}

