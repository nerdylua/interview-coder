// ProcessingHelper.ts
import fs from "node:fs"
import path from "node:path"
import { ScreenshotHelper } from "./ScreenshotHelper"
import { IProcessingHelperDeps } from "./main"
import * as axios from "axios"
import { app, BrowserWindow, dialog } from "electron"
import { OpenAI } from "openai"
import { configHelper } from "./ConfigHelper"
import Anthropic from '@anthropic-ai/sdk';

// Interface for Gemini API requests
interface GeminiMessage {
  role: string;
  parts: Array<{
    text?: string;
    inlineData?: {
      mimeType: string;
      data: string;
    }
  }>;
}

interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{
        text: string;
      }>;
    };
    finishReason: string;
  }>;
}
interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: Array<{
    type: 'text' | 'image';
    text?: string;
    source?: {
      type: 'base64';
      media_type: string;
      data: string;
    };
  }>;
}
export class ProcessingHelper {
  private deps: IProcessingHelperDeps
  private screenshotHelper: ScreenshotHelper
  private openaiClient: OpenAI | null = null
  private geminiApiKey: string | null = null
  private anthropicClient: Anthropic | null = null

  // AbortControllers for API requests
  private currentProcessingAbortController: AbortController | null = null
  private currentExtraProcessingAbortController: AbortController | null = null

  constructor(deps: IProcessingHelperDeps) {
    this.deps = deps
    this.screenshotHelper = deps.getScreenshotHelper()
    
    // Initialize AI client based on config
    this.initializeAIClient();
    
    // Listen for config changes to re-initialize the AI client
    configHelper.on('config-updated', () => {
      this.initializeAIClient();
    });
  }
  
  /**
   * Initialize or reinitialize the AI client with current config
   */
  private initializeAIClient(): void {
    try {
      const config = configHelper.loadConfig();
      
      if (config.apiProvider === "openai") {
        if (config.apiKey) {
          this.openaiClient = new OpenAI({ 
            apiKey: config.apiKey,
            timeout: 60000, // 60 second timeout
            maxRetries: 2   // Retry up to 2 times
          });
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.log("OpenAI client initialized successfully");
        } else {
          this.openaiClient = null;
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.warn("No API key available, OpenAI client not initialized");
        }
      } else if (config.apiProvider === "gemini"){
        // Gemini client initialization
        this.openaiClient = null;
        this.anthropicClient = null;
        if (config.apiKey) {
          this.geminiApiKey = config.apiKey;
          console.log("Gemini API key set successfully");
        } else {
          this.openaiClient = null;
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.warn("No API key available, Gemini client not initialized");
        }
      } else if (config.apiProvider === "anthropic") {
        // Reset other clients
        this.openaiClient = null;
        this.geminiApiKey = null;
        if (config.apiKey) {
          this.anthropicClient = new Anthropic({
            apiKey: config.apiKey,
            timeout: 60000,
            maxRetries: 2
          });
          console.log("Anthropic client initialized successfully");
        } else {
          this.openaiClient = null;
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.warn("No API key available, Anthropic client not initialized");
        }
      }
    } catch (error) {
      console.error("Failed to initialize AI client:", error);
      this.openaiClient = null;
      this.geminiApiKey = null;
      this.anthropicClient = null;
    }
  }

  private async waitForInitialization(
    mainWindow: BrowserWindow
  ): Promise<void> {
    let attempts = 0
    const maxAttempts = 50 // 5 seconds total

    while (attempts < maxAttempts) {
      const isInitialized = await mainWindow.webContents.executeJavaScript(
        "window.__IS_INITIALIZED__"
      )
      if (isInitialized) return
      await new Promise((resolve) => setTimeout(resolve, 100))
      attempts++
    }
    throw new Error("App failed to initialize after 5 seconds")
  }

  private async getCredits(): Promise<number> {
    const mainWindow = this.deps.getMainWindow()
    if (!mainWindow) return 999 // Unlimited credits in this version

    try {
      await this.waitForInitialization(mainWindow)
      return 999 // Always return sufficient credits to work
    } catch (error) {
      console.error("Error getting credits:", error)
      return 999 // Unlimited credits as fallback
    }
  }

  private async getLanguage(): Promise<string> {
    try {
      // Get language from config
      const config = configHelper.loadConfig();
      if (config.language) {
        return config.language;
      }
      
      // Fallback to window variable if config doesn't have language
      const mainWindow = this.deps.getMainWindow()
      if (mainWindow) {
        try {
          await this.waitForInitialization(mainWindow)
          const language = await mainWindow.webContents.executeJavaScript(
            "window.__LANGUAGE__"
          )

          if (
            typeof language === "string" &&
            language !== undefined &&
            language !== null
          ) {
            return language;
          }
        } catch (err) {
          console.warn("Could not get language from window", err);
        }
      }
      
      // Default fallback
      return "python";
    } catch (error) {
      console.error("Error getting language:", error)
      return "python"
    }
  }

  public async processScreenshots(): Promise<void> {
    const mainWindow = this.deps.getMainWindow()
    if (!mainWindow) return

    const config = configHelper.loadConfig();
    
    // First verify we have a valid AI client
    if (config.apiProvider === "openai" && !this.openaiClient) {
      this.initializeAIClient();
      
      if (!this.openaiClient) {
        console.error("OpenAI client not initialized");
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.API_KEY_INVALID
        );
        return;
      }
    } else if (config.apiProvider === "gemini" && !this.geminiApiKey) {
      this.initializeAIClient();
      
      if (!this.geminiApiKey) {
        console.error("Gemini API key not initialized");
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.API_KEY_INVALID
        );
        return;
      }
    } else if (config.apiProvider === "anthropic" && !this.anthropicClient) {
      // Add check for Anthropic client
      this.initializeAIClient();
      
      if (!this.anthropicClient) {
        console.error("Anthropic client not initialized");
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.API_KEY_INVALID
        );
        return;
      }
    }

    const view = this.deps.getView()
    const appMode = this.deps.getAppMode()
    console.log("Processing screenshots in view:", view, "mode:", appMode)

    if (view === "queue") {
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_START)
      const screenshotQueue = this.screenshotHelper.getScreenshotQueue()
      console.log("Processing main queue screenshots:", screenshotQueue)
      
      // Check if the queue is empty
      if (!screenshotQueue || screenshotQueue.length === 0) {
        console.log("No screenshots found in queue");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }

      // Check that files actually exist
      const existingScreenshots = screenshotQueue.filter(path => fs.existsSync(path));
      if (existingScreenshots.length === 0) {
        console.log("Screenshot files don't exist on disk");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }

      try {
        // Initialize AbortController
        this.currentProcessingAbortController = new AbortController()
        const { signal } = this.currentProcessingAbortController

        const screenshots = await Promise.all(
          existingScreenshots.map(async (path) => {
            try {
              return {
                path,
                preview: await this.screenshotHelper.getImagePreview(path),
                data: fs.readFileSync(path).toString('base64')
              };
            } catch (err) {
              console.error(`Error reading screenshot ${path}:`, err);
              return null;
            }
          })
        )

        // Filter out any nulls from failed screenshots
        const validScreenshots = screenshots.filter(Boolean);
        
        if (validScreenshots.length === 0) {
          throw new Error("Failed to load screenshot data");
        }

        const result = appMode === "coding"
          ? await this.processScreenshotsHelper(validScreenshots, signal)
          : await this.processNonCodingScreenshots(validScreenshots, signal)

        if (!result.success) {
          console.log("Processing failed:", result.error)
          if (result.error?.includes("API Key") || result.error?.includes("OpenAI") || result.error?.includes("Gemini")) {
            mainWindow.webContents.send(
              this.deps.PROCESSING_EVENTS.API_KEY_INVALID
            )
          } else {
            mainWindow.webContents.send(
              this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR,
              result.error
            )
          }
          // Reset view back to queue on error
          console.log("Resetting view to queue due to error")
          this.deps.setView("queue")
          return
        }

        // Only set view to solutions if processing succeeded
        console.log("Setting view to solutions after successful processing")
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.SOLUTION_SUCCESS,
          result.data
        )
        this.deps.setView("solutions")
      } catch (error: any) {
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR,
          error
        )
        console.error("Processing error:", error)
        if (axios.isCancel(error)) {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR,
            "Processing was canceled by the user."
          )
        } else {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR,
            error.message || "Server error. Please try again."
          )
        }
        // Reset view back to queue on error
        console.log("Resetting view to queue due to error")
        this.deps.setView("queue")
      } finally {
        this.currentProcessingAbortController = null
      }
    } else {
      // view == 'solutions'

      // For non-coding mode, don't allow debug mode
      if (appMode === "non-coding") {
        console.log("Debug mode not available in non-coding mode");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }

      const extraScreenshotQueue =
        this.screenshotHelper.getExtraScreenshotQueue()
      console.log("Processing extra queue screenshots:", extraScreenshotQueue)

      // Check if the extra queue is empty
      if (!extraScreenshotQueue || extraScreenshotQueue.length === 0) {
        console.log("No extra screenshots found in queue");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);

        return;
      }

      // Check that files actually exist
      const existingExtraScreenshots = extraScreenshotQueue.filter(path => fs.existsSync(path));
      if (existingExtraScreenshots.length === 0) {
        console.log("Extra screenshot files don't exist on disk");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }
      
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_START)

      // Initialize AbortController
      this.currentExtraProcessingAbortController = new AbortController()
      const { signal } = this.currentExtraProcessingAbortController

      try {
        // Get all screenshots (both main and extra) for processing
        const allPaths = [
          ...this.screenshotHelper.getScreenshotQueue(),
          ...existingExtraScreenshots
        ];
        
        const screenshots = await Promise.all(
          allPaths.map(async (path) => {
            try {
              if (!fs.existsSync(path)) {
                console.warn(`Screenshot file does not exist: ${path}`);
                return null;
              }
              
              return {
                path,
                preview: await this.screenshotHelper.getImagePreview(path),
                data: fs.readFileSync(path).toString('base64')
              };
            } catch (err) {
              console.error(`Error reading screenshot ${path}:`, err);
              return null;
            }
          })
        )
        
        // Filter out any nulls from failed screenshots
        const validScreenshots = screenshots.filter(Boolean);
        
        if (validScreenshots.length === 0) {
          throw new Error("Failed to load screenshot data for debugging");
        }
        
        console.log(
          "Combined screenshots for processing:",
          validScreenshots.map((s) => s.path)
        )

        const result = await this.processExtraScreenshotsHelper(
          validScreenshots,
          signal
        )

        if (result.success) {
          this.deps.setHasDebugged(true)
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.DEBUG_SUCCESS,
            result.data
          )
        } else {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.DEBUG_ERROR,
            result.error
          )
        }
      } catch (error: any) {
        if (axios.isCancel(error)) {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.DEBUG_ERROR,
            "Extra processing was canceled by the user."
          )
        } else {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.DEBUG_ERROR,
            error.message
          )
        }
      } finally {
        this.currentExtraProcessingAbortController = null
      }
    }
  }

  private async processScreenshotsHelper(
    screenshots: Array<{ path: string; data: string }>,
    signal: AbortSignal
  ) {
    try {
      const config = configHelper.loadConfig();
      const language = await this.getLanguage();
      const mainWindow = this.deps.getMainWindow();
      
      // Step 1: Extract problem info using AI Vision API (OpenAI or Gemini)
      const imageDataList = screenshots.map(screenshot => screenshot.data);
      
      // Update the user on progress
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Analyzing problem from screenshots...",
          progress: 20
        });
      }

      let problemInfo;
      
      if (config.apiProvider === "openai") {
        // Verify OpenAI client
        if (!this.openaiClient) {
          this.initializeAIClient(); // Try to reinitialize
          
          if (!this.openaiClient) {
            return {
              success: false,
              error: "OpenAI API key not configured or invalid. Please check your settings."
            };
          }
        }

        // Use OpenAI for processing
        const messages = [
          {
            role: "system" as const,
            content: "You are a coding challenge interpreter. Carefully analyze the screenshot(s) to extract the complete coding problem. Focus on capturing the EXACT problem statement as written, including all details, requirements, and nuances. Return the information in JSON format with these fields: problem_statement, constraints, example_input, example_output. Just return the structured JSON without any other text."
          },
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const,
                text: `Extract the complete coding problem details from these screenshots. Pay special attention to:
1. The EXACT problem statement - capture every detail, requirement, and specification as written
2. All constraints (time limits, space limits, input ranges, etc.)
3. Complete example inputs and outputs with proper formatting
4. Any special conditions, edge cases, or additional requirements mentioned

Return in JSON format. Preferred coding language: ${language}.`
              },
              ...imageDataList.map(data => ({
                type: "image_url" as const,
                image_url: { url: `data:image/png;base64,${data}` }
              }))
            ]
          }
        ];

        // Send to OpenAI Vision API
        const extractionResponse = await this.openaiClient.chat.completions.create({
          model: config.extractionModel || "gpt-4.1",
          messages: messages,
          max_tokens: 4000,
          temperature: 0.2
        });

        // Parse the response
        try {
          const responseText = extractionResponse.choices[0].message.content;
          // Handle when OpenAI might wrap the JSON in markdown code blocks
          const jsonText = responseText.replace(/```json|```/g, '').trim();
          problemInfo = JSON.parse(jsonText);
        } catch (error) {
          console.error("Error parsing OpenAI response:", error);
          return {
            success: false,
            error: "Failed to parse problem information. Please try again or use clearer screenshots."
          };
        }
      } else if (config.apiProvider === "gemini")  {
        // Use Gemini API
        if (!this.geminiApiKey) {
          return {
            success: false,
            error: "Gemini API key not configured. Please check your settings."
          };
        }

        try {
          // Create Gemini message structure
          const geminiMessages: GeminiMessage[] = [
            {
              role: "user",
              parts: [
                {
                  text: `You are a coding challenge interpreter. Carefully analyze the screenshots to extract the complete coding problem. Focus on capturing the EXACT problem statement as written, including all details, requirements, and nuances.

Pay special attention to:
1. The EXACT problem statement - capture every detail, requirement, and specification as written
2. All constraints (time limits, space limits, input ranges, etc.)
3. Complete example inputs and outputs with proper formatting
4. Any special conditions, edge cases, or additional requirements mentioned

Return the information in JSON format with these fields: problem_statement, constraints, example_input, example_output. Just return the structured JSON without any other text. Preferred coding language: ${language}.`
                },
                ...imageDataList.map(data => ({
                  inlineData: {
                    mimeType: "image/png",
                    data: data
                  }
                }))
              ]
            }
          ];

          // Make API request to Gemini
          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/${config.extractionModel || "gemini-2.5-flash"}:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 8000  // Increased for 2.5 models
              }
            },
            { signal }
          );

          const responseData = response.data as GeminiResponse;

          console.log("Gemini extraction API response structure:", JSON.stringify(responseData, null, 2));

          if (!responseData.candidates || responseData.candidates.length === 0) {
            console.error("No candidates in Gemini extraction response:", responseData);
            throw new Error("Empty response from Gemini API");
          }

          const candidate = responseData.candidates[0];
          if (!candidate || !candidate.content) {
            console.error("Invalid candidate structure in extraction:", candidate);
            throw new Error("Invalid response structure from Gemini API");
          }

          // Handle MAX_TOKENS case
          if (candidate.finishReason === "MAX_TOKENS") {
            console.error("Gemini response hit token limit. Consider using gemini-2.5-flash instead of pro for extraction.");
            throw new Error("Response was truncated due to token limits. Try using Gemini 2.5 Flash model instead.");
          }

          if (!candidate.content.parts || candidate.content.parts.length === 0) {
            console.error("No parts in candidate content:", candidate.content);
            throw new Error("No content parts in Gemini response");
          }

          const responseText = candidate.content.parts[0].text;
          
          // Handle when Gemini might wrap the JSON in markdown code blocks
          const jsonText = responseText.replace(/```json|```/g, '').trim();
          problemInfo = JSON.parse(jsonText);
        } catch (error) {
          console.error("Error using Gemini API:", error);
          return {
            success: false,
            error: "Failed to process with Gemini API. Please check your API key or try again later."
          };
        }
      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return {
            success: false,
            error: "Anthropic API key not configured. Please check your settings."
          };
        }

        try {
          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: `Extract the complete coding problem details from these screenshots. Pay special attention to:
1. The EXACT problem statement - capture every detail, requirement, and specification as written
2. All constraints (time limits, space limits, input ranges, etc.)
3. Complete example inputs and outputs with proper formatting
4. Any special conditions, edge cases, or additional requirements mentioned

Return in JSON format with these fields: problem_statement, constraints, example_input, example_output. Preferred coding language: ${language}.`
                },
                ...imageDataList.map(data => ({
                  type: "image" as const,
                  source: {
                    type: "base64" as const,
                    media_type: "image/png" as const,
                    data: data
                  }
                }))
              ]
            }
          ];

          const response = await this.anthropicClient.messages.create({
            model: config.extractionModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages: messages,
            temperature: 0.2
          });

          const responseText = (response.content[0] as { type: 'text', text: string }).text;
          const jsonText = responseText.replace(/```json|```/g, '').trim();
          problemInfo = JSON.parse(jsonText);
        } catch (error: any) {
          console.error("Error using Anthropic API:", error);

          // Add specific handling for Claude's limitations
          if (error.status === 429) {
            return {
              success: false,
              error: "Claude API rate limit exceeded. Please wait a few minutes before trying again."
            };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return {
              success: false,
              error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs."
            };
          }

          return {
            success: false,
            error: "Failed to process with Anthropic API. Please check your API key or try again later."
          };
        }
      }
      
      // Update the user on progress
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Problem analyzed successfully. Preparing to generate solution...",
          progress: 40
        });
      }

      // Store problem info in AppState
      this.deps.setProblemInfo(problemInfo);

      // Send first success event
      if (mainWindow) {
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.PROBLEM_EXTRACTED,
          problemInfo
        );

        // Generate solutions after successful extraction
        const solutionsResult = await this.generateSolutionsHelper(signal);
        if (solutionsResult.success) {
          // Clear any existing extra screenshots before transitioning to solutions view
          this.screenshotHelper.clearExtraScreenshotQueue();
          
          // Final progress update
          mainWindow.webContents.send("processing-status", {
            message: "Solution generated successfully",
            progress: 100
          });
          
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.SOLUTION_SUCCESS,
            solutionsResult.data
          );
          return { success: true, data: solutionsResult.data };
        } else {
          throw new Error(
            solutionsResult.error || "Failed to generate solutions"
          );
        }
      }

      return { success: false, error: "Failed to process screenshots" };
    } catch (error: any) {
      // If the request was cancelled, don't retry
      if (axios.isCancel(error)) {
        return {
          success: false,
          error: "Processing was canceled by the user."
        };
      }
      
      // Handle OpenAI API errors specifically
      if (error?.response?.status === 401) {
        return {
          success: false,
          error: "Invalid OpenAI API key. Please check your settings."
        };
      } else if (error?.response?.status === 429) {
        return {
          success: false,
          error: "OpenAI API rate limit exceeded or insufficient credits. Please try again later."
        };
      } else if (error?.response?.status === 500) {
        return {
          success: false,
          error: "OpenAI server error. Please try again later."
        };
      }

      console.error("API Error Details:", error);
      return { 
        success: false, 
        error: error.message || "Failed to process screenshots. Please try again." 
      };
    }
  }

  private async generateSolutionsHelper(signal: AbortSignal) {
    try {
      const problemInfo = this.deps.getProblemInfo();
      const language = await this.getLanguage();
      const config = configHelper.loadConfig();
      const mainWindow = this.deps.getMainWindow();

      if (!problemInfo) {
        throw new Error("No problem info available");
      }

      // Update progress status
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Creating optimal solution with detailed explanations...",
          progress: 60
        });
      }

      // Create prompt for solution generation
      const promptText = `
Generate a detailed solution for the following coding problem:

PROBLEM STATEMENT:
${problemInfo.problem_statement}

CONSTRAINTS:
${problemInfo.constraints || "No specific constraints provided."}

EXAMPLE INPUT:
${problemInfo.example_input || "No example input provided."}

EXAMPLE OUTPUT:
${problemInfo.example_output || "No example output provided."}

LANGUAGE: ${language}

I need the response in the following format:

**Code:**
\`\`\`${language.toLowerCase()}
// Your solution here
\`\`\`

**My Thoughts:**
Write this as if you're explaining your thought process to a friend. Be conversational and natural, sharing your reasoning, key insights, and why you chose this approach. Include 3-5 thoughtful points about your strategy, any tradeoffs you considered, and what makes this solution effective. Use bullet points or numbered list format.

**Time complexity:** O(X) - Provide a crisp, clear explanation in 1-2 sentences that explains both what the complexity is and why it's that complexity.

**Space complexity:** O(X) - Provide a crisp, clear explanation in 1-2 sentences that explains both what the complexity is and why it's that complexity.

For the "My Thoughts" section, write naturally as if you're having a conversation. Use this format:
- I immediately thought about using [approach] because [reason]
- The key insight is that we can [insight]
- What's clever about this approach is [explanation]
- I considered [alternative] first, but realized [why current approach is better]
- This solution handles [edge cases/benefits]

For complexity explanations, be concise but complete. Examples:
- "Time complexity: O(n) - We scan through the array once, and each hash map operation is O(1)"
- "Space complexity: O(n) - In the worst case, we store all n elements in our hash map"

Your solution should be efficient, well-commented, and handle edge cases.
`;

      let responseContent;
      
      if (config.apiProvider === "openai") {
        // OpenAI processing
        if (!this.openaiClient) {
          return {
            success: false,
            error: "OpenAI API key not configured. Please check your settings."
          };
        }
        
        // Send to OpenAI API
        const solutionResponse = await this.openaiClient.chat.completions.create({
          model: config.solutionModel || "gpt-4.1",
          messages: [
            { role: "system", content: "You are an expert coding interview assistant. Provide clear, optimal solutions with detailed explanations. Write your thoughts in a conversational, human-like style as if explaining to a friend. Be crisp and clear with complexity analysis." },
            { role: "user", content: promptText }
          ],
          max_tokens: 4000,
          temperature: 0.2
        });

        responseContent = solutionResponse.choices[0].message.content;
      } else if (config.apiProvider === "gemini")  {
        // Gemini processing
        if (!this.geminiApiKey) {
          return {
            success: false,
            error: "Gemini API key not configured. Please check your settings."
          };
        }
        
        try {
          // Create Gemini message structure
          const geminiMessages = [
            {
              role: "user",
              parts: [
                {
                  text: `You are an expert coding interview assistant. Provide a clear, optimal solution with detailed explanations. Write your thoughts in a conversational, human-like style as if explaining to a friend. Be crisp and clear with complexity analysis.\n\n${promptText}`
                }
              ]
            }
          ];

          // Make API request to Gemini
          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/${config.solutionModel || "gemini-2.5-flash"}:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 8000  // Increased for 2.5 models
              }
            },
            { signal }
          );

          const responseData = response.data as GeminiResponse;

          console.log("Gemini API response structure:", JSON.stringify(responseData, null, 2));

          if (!responseData.candidates || responseData.candidates.length === 0) {
            console.error("No candidates in Gemini response:", responseData);
            throw new Error("Empty response from Gemini API");
          }

          const candidate = responseData.candidates[0];
          if (!candidate) {
            console.error("First candidate is undefined:", responseData);
            throw new Error("Invalid candidate in Gemini response");
          }

          // Handle MAX_TOKENS case
          if (candidate.finishReason === "MAX_TOKENS") {
            console.error("Gemini solution response hit token limit. Consider using gemini-2.5-flash instead of pro.");
            throw new Error("Response was truncated due to token limits. Try using Gemini 2.5 Flash model instead.");
          }

          if (!candidate.content) {
            console.error("Candidate content is undefined:", candidate);
            throw new Error("No content in Gemini candidate");
          }

          if (!candidate.content.parts || candidate.content.parts.length === 0) {
            console.error("Candidate content parts is undefined or empty:", candidate.content);
            throw new Error("No parts in Gemini candidate content");
          }

          const firstPart = candidate.content.parts[0];
          if (!firstPart || !firstPart.text) {
            console.error("First part or text is undefined:", firstPart);
            throw new Error("No text in Gemini candidate first part");
          }

          responseContent = firstPart.text;
        } catch (error) {
          console.error("Error using Gemini API for solution:", error);
          return {
            success: false,
            error: "Failed to generate solution with Gemini API. Please check your API key or try again later."
          };
        }
      } else if (config.apiProvider === "anthropic") {
        // Anthropic processing
        if (!this.anthropicClient) {
          return {
            success: false,
            error: "Anthropic API key not configured. Please check your settings."
          };
        }
        
        try {
          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: `You are an expert coding interview assistant. Provide a clear, optimal solution with detailed explanations. Write your thoughts in a conversational, human-like style as if explaining to a friend. Be crisp and clear with complexity analysis.\n\n${promptText}`
                }
              ]
            }
          ];

          // Send to Anthropic API
          const response = await this.anthropicClient.messages.create({
            model: config.solutionModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages: messages,
            temperature: 0.2
          });

          responseContent = (response.content[0] as { type: 'text', text: string }).text;
        } catch (error: any) {
          console.error("Error using Anthropic API for solution:", error);

          // Add specific handling for Claude's limitations
          if (error.status === 429) {
            return {
              success: false,
              error: "Claude API rate limit exceeded. Please wait a few minutes before trying again."
            };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return {
              success: false,
              error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs."
            };
          }

          return {
            success: false,
            error: "Failed to generate solution with Anthropic API. Please check your API key or try again later."
          };
        }
      }
      
      // Extract parts from the response
      const codeMatch = responseContent.match(/```(?:\w+)?\s*([\s\S]*?)```/);
      const code = codeMatch ? codeMatch[1].trim() : responseContent;
      
      // Extract thoughts with improved regex and fallback logic
      let thoughts: string[] = [];

      // Try multiple patterns to extract thoughts
      const thoughtsPatterns = [
        /(?:My Thoughts:|Thoughts:|Key Insights:|Reasoning:|Approach:)\s*([\s\S]*?)(?:\n\s*(?:Time complexity|Space complexity|Complexity|Code:|```|\d+\.|$))/i,
        /(?:My Thoughts:|Thoughts:|Key Insights:|Reasoning:|Approach:)\s*([\s\S]*?)(?=\n\s*[A-Z][a-z]+\s*complexity|$)/i,
        /(?:My Thoughts:|Thoughts:|Key Insights:|Reasoning:|Approach:)\s*([\s\S]*?)(?=\n\s*\d+\.|$)/i
      ];

      let thoughtsContent = "";
      for (const pattern of thoughtsPatterns) {
        const match = responseContent.match(pattern);
        if (match && match[1] && match[1].trim().length > 5) {
          thoughtsContent = match[1].trim();
          break;
        }
      }

      if (thoughtsContent) {
        // Extract bullet points or numbered items
        const bulletPoints = thoughtsContent.match(/(?:^|\n)\s*(?:[-*•]|\d+\.)\s*([^\n]+)/g);
        if (bulletPoints && bulletPoints.length > 0) {
          thoughts = bulletPoints.map(point =>
            point.replace(/^\s*(?:[-*•]|\d+\.)\s*/, '').trim()
          ).filter(point => point.length > 3); // Filter out very short points
        } else {
          // If no bullet points, try to extract meaningful sentences
          const sentences = thoughtsContent.split(/[.!?]+/)
            .map(sentence => sentence.trim())
            .filter(sentence => sentence.length > 10 && !sentence.match(/^(Time|Space)\s+complexity/i));

          if (sentences.length > 0) {
            // Take first 3-5 meaningful sentences and format them as thoughts
            thoughts = sentences.slice(0, 5).map(sentence => {
              // Ensure sentence ends with proper punctuation
              if (!sentence.match(/[.!?]$/)) {
                sentence += '.';
              }
              return sentence;
            });
          }
        }
      }

      // Fallback: if still no thoughts found, create default conversational thoughts
      if (thoughts.length === 0) {
        console.warn("No thoughts extracted from response, using fallback thoughts");
        console.log("Response content for debugging:", responseContent.substring(0, 500));
        thoughts = [
          "I approached this problem by analyzing the requirements and identifying the most efficient solution pattern.",
          "The key insight is choosing the right data structure and algorithm to optimize both time and space complexity.",
          "This solution handles edge cases well and maintains clean, readable code structure."
        ];
      } else {
        console.log(`Successfully extracted ${thoughts.length} thoughts from response`);
      }
      
      // Extract complexity information
      const timeComplexityPattern = /Time complexity:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*(?:Space complexity|$))/i;
      const spaceComplexityPattern = /Space complexity:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*(?:[A-Z]|$))/i;
      
      let timeComplexity = "O(n) - We scan through the array once, and each hash map operation is O(1)";
      let spaceComplexity = "O(n) - In the worst case, we store all n elements in our hash map";
      
      const timeMatch = responseContent.match(timeComplexityPattern);
      if (timeMatch && timeMatch[1]) {
        timeComplexity = timeMatch[1].trim();
        if (!timeComplexity.match(/O\([^)]+\)/i)) {
          timeComplexity = `O(n) - ${timeComplexity}`;
        } else if (!timeComplexity.includes('-') && !timeComplexity.includes('because')) {
          const notationMatch = timeComplexity.match(/O\([^)]+\)/i);
          if (notationMatch) {
            const notation = notationMatch[0];
            const rest = timeComplexity.replace(notation, '').trim();
            timeComplexity = `${notation} - ${rest}`;
          }
        }
      }
      
      const spaceMatch = responseContent.match(spaceComplexityPattern);
      if (spaceMatch && spaceMatch[1]) {
        spaceComplexity = spaceMatch[1].trim();
        if (!spaceComplexity.match(/O\([^)]+\)/i)) {
          spaceComplexity = `O(n) - ${spaceComplexity}`;
        } else if (!spaceComplexity.includes('-') && !spaceComplexity.includes('because')) {
          const notationMatch = spaceComplexity.match(/O\([^)]+\)/i);
          if (notationMatch) {
            const notation = notationMatch[0];
            const rest = spaceComplexity.replace(notation, '').trim();
            spaceComplexity = `${notation} - ${rest}`;
          }
        }
      }

      const formattedResponse = {
        code: code,
        thoughts: thoughts.length > 0 ? thoughts : ["Solution approach based on efficiency and readability"],
        time_complexity: timeComplexity,
        space_complexity: spaceComplexity
      };

      return { success: true, data: formattedResponse };
    } catch (error: any) {
      if (axios.isCancel(error)) {
        return {
          success: false,
          error: "Processing was canceled by the user."
        };
      }
      
      if (error?.response?.status === 401) {
        return {
          success: false,
          error: "Invalid OpenAI API key. Please check your settings."
        };
      } else if (error?.response?.status === 429) {
        return {
          success: false,
          error: "OpenAI API rate limit exceeded or insufficient credits. Please try again later."
        };
      }
      
      console.error("Solution generation error:", error);
      return { success: false, error: error.message || "Failed to generate solution" };
    }
  }

  private async processExtraScreenshotsHelper(
    screenshots: Array<{ path: string; data: string }>,
    signal: AbortSignal
  ) {
    try {
      const problemInfo = this.deps.getProblemInfo();
      const language = await this.getLanguage();
      const config = configHelper.loadConfig();
      const mainWindow = this.deps.getMainWindow();

      if (!problemInfo) {
        throw new Error("No problem info available");
      }

      // Update progress status
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Processing debug screenshots...",
          progress: 30
        });
      }

      // Prepare the images for the API call
      const imageDataList = screenshots.map(screenshot => screenshot.data);
      
      let debugContent;
      
      if (config.apiProvider === "openai") {
        if (!this.openaiClient) {
          return {
            success: false,
            error: "OpenAI API key not configured. Please check your settings."
          };
        }
        
        const messages = [
          {
            role: "system" as const, 
            content: `You are a coding interview assistant helping debug and improve solutions. Analyze these screenshots which include either error messages, incorrect outputs, or test cases, and provide detailed debugging help.

CRITICAL: Your response MUST include a complete, corrected code solution in a markdown code block at the beginning, even if the changes are minor.

Your response MUST follow this exact structure with these section headers (use ### for headers):
### Corrected Code
\`\`\`language
// Always provide the complete, corrected solution here
\`\`\`

### Issues Found
- Concise bullet points of specific problems

### Changes Made
- Brief bullet points of what was fixed

### Complexity
Time: O(X) - Brief explanation
Space: O(X) - Brief explanation

### Key Points
- Most important takeaways

If you include additional code examples, use proper markdown code blocks with language specification (e.g. \`\`\`java).`
          },
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const, 
                text: `I'm solving this coding problem: "${problemInfo.problem_statement}" in ${language}. I need help with debugging or improving my solution. Here are screenshots of my code, the errors or test cases. Please provide a detailed analysis with:
1. What issues you found in my code
2. Specific improvements and corrections
3. Any optimizations that would make the solution better
4. A clear explanation of the changes needed` 
              },
              ...imageDataList.map(data => ({
                type: "image_url" as const,
                image_url: { url: `data:image/png;base64,${data}` }
              }))
            ]
          }
        ];

        if (mainWindow) {
          mainWindow.webContents.send("processing-status", {
            message: "Analyzing code and generating debug feedback...",
            progress: 60
          });
        }

        const debugResponse = await this.openaiClient.chat.completions.create({
          model: config.debuggingModel || "gpt-4.1",
          messages: messages,
          max_tokens: 4000,
          temperature: 0.2
        });
        
        debugContent = debugResponse.choices[0].message.content;
      } else if (config.apiProvider === "gemini")  {
        if (!this.geminiApiKey) {
          return {
            success: false,
            error: "Gemini API key not configured. Please check your settings."
          };
        }
        
        try {
          const debugPrompt = `
You are a coding interview assistant helping debug and improve solutions. Analyze these screenshots which include either error messages, incorrect outputs, or test cases, and provide detailed debugging help.

I'm solving this coding problem: "${problemInfo.problem_statement}" in ${language}. I need help with debugging or improving my solution.

CRITICAL: Your response MUST include a complete, corrected code solution in a markdown code block at the beginning, even if the changes are minor.

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE WITH THESE SECTION HEADERS:
### Corrected Code
\`\`\`${language.toLowerCase()}
// Always provide the complete, corrected solution here
\`\`\`

### Issues Found
- Concise bullet points of specific problems

### Changes Made
- Brief bullet points of what was fixed

### Complexity
Time: O(X) - Brief explanation
Space: O(X) - Brief explanation

### Key Points
- Most important takeaways

If you include additional code examples, use proper markdown code blocks with language specification (e.g. \`\`\`java).
`;

          const geminiMessages = [
            {
              role: "user",
              parts: [
                { text: debugPrompt },
                ...imageDataList.map(data => ({
                  inlineData: {
                    mimeType: "image/png",
                    data: data
                  }
                }))
              ]
            }
          ];

          if (mainWindow) {
            mainWindow.webContents.send("processing-status", {
              message: "Analyzing code and generating debug feedback with Gemini...",
              progress: 60
            });
          }

          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/${config.debuggingModel || "gemini-2.5-flash"}:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 8000  // Increased for 2.5 models
              }
            },
            { signal }
          );

          const responseData = response.data as GeminiResponse;
          
          if (!responseData.candidates || responseData.candidates.length === 0) {
            throw new Error("Empty response from Gemini API");
          }
          
          debugContent = responseData.candidates[0].content.parts[0].text;
        } catch (error) {
          console.error("Error using Gemini API for debugging:", error);
          return {
            success: false,
            error: "Failed to process debug request with Gemini API. Please check your API key or try again later."
          };
        }
      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return {
            success: false,
            error: "Anthropic API key not configured. Please check your settings."
          };
        }
        
        try {
          const debugPrompt = `
You are a coding interview assistant helping debug and improve solutions. Analyze these screenshots which include either error messages, incorrect outputs, or test cases, and provide detailed debugging help.

I'm solving this coding problem: "${problemInfo.problem_statement}" in ${language}. I need help with debugging or improving my solution.

CRITICAL: Your response MUST include a complete, corrected code solution in a markdown code block at the beginning, even if the changes are minor.

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE WITH THESE SECTION HEADERS:
### Corrected Code
\`\`\`${language.toLowerCase()}
// Always provide the complete, corrected solution here
\`\`\`

### Issues Found
- Concise bullet points of specific problems

### Changes Made
- Brief bullet points of what was fixed

### Complexity
Time: O(X) - Brief explanation
Space: O(X) - Brief explanation

### Key Points
- Most important takeaways

If you include additional code examples, use proper markdown code blocks with language specification.
`;

          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: debugPrompt
                },
                ...imageDataList.map(data => ({
                  type: "image" as const,
                  source: {
                    type: "base64" as const,
                    media_type: "image/png" as const, 
                    data: data
                  }
                }))
              ]
            }
          ];

          if (mainWindow) {
            mainWindow.webContents.send("processing-status", {
              message: "Analyzing code and generating debug feedback with Claude...",
              progress: 60
            });
          }

          const response = await this.anthropicClient.messages.create({
            model: config.debuggingModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages: messages,
            temperature: 0.2
          });
          
          debugContent = (response.content[0] as { type: 'text', text: string }).text;
        } catch (error: any) {
          console.error("Error using Anthropic API for debugging:", error);
          
          // Add specific handling for Claude's limitations
          if (error.status === 429) {
            return {
              success: false,
              error: "Claude API rate limit exceeded. Please wait a few minutes before trying again."
            };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return {
              success: false,
              error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs."
            };
          }
          
          return {
            success: false,
            error: "Failed to process debug request with Anthropic API. Please check your API key or try again later."
          };
        }
      }
      
      
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Debug analysis complete",
          progress: 100
        });
      }

      // Extract code from debug content - ensure we always have code
      let extractedCode = "";
      const codeMatch = debugContent.match(/```(?:[a-zA-Z]+)?([\s\S]*?)```/);
      if (codeMatch && codeMatch[1]) {
        extractedCode = codeMatch[1].trim();
      }

      // If no code was found in the response, get the original problem info and generate a basic solution
      if (!extractedCode || extractedCode.length < 10) {
        const problemInfo = this.deps.getProblemInfo();
        const language = await this.getLanguage();

        if (problemInfo) {
          extractedCode = `// ${language} solution for: ${problemInfo.problem_statement?.substring(0, 100)}...
// Please refer to the analysis below for specific improvements needed

// Basic structure - modify based on analysis:
function solution() {
    // TODO: Implement based on debug analysis
    return result;
}`;
        } else {
          extractedCode = `// Debug analysis provided below
// Please implement the solution based on the feedback

function solution() {
    // TODO: Implement based on debug analysis
    return result;
}`;
        }
      }

      // Remove the code block from debug content to avoid duplication
      let formattedDebugContent = debugContent;

      // Remove the first code block (which is the corrected code) from the analysis
      const codeBlockRegex = /```(?:[a-zA-Z]+)?([\s\S]*?)```/;
      const firstCodeBlockMatch = formattedDebugContent.match(codeBlockRegex);
      if (firstCodeBlockMatch) {
        formattedDebugContent = formattedDebugContent.replace(codeBlockRegex, '').trim();
      }

      // Remove the "Corrected Code" section header if it exists
      formattedDebugContent = formattedDebugContent.replace(/###\s*Corrected Code\s*\n?/i, '').trim();

      if (!formattedDebugContent.includes('# ') && !formattedDebugContent.includes('## ')) {
        formattedDebugContent = formattedDebugContent
          .replace(/issues identified|problems found|bugs found/i, '## Issues Found')
          .replace(/code improvements|improvements|suggested changes/i, '## Changes Made')
          .replace(/optimizations|performance improvements/i, '## Optimizations')
          .replace(/explanation|detailed analysis/i, '## Explanation');
      }

      const bulletPoints = formattedDebugContent.match(/(?:^|\n)[ ]*(?:[-*•]|\d+\.)[ ]+([^\n]+)/g);
      const thoughts = bulletPoints 
        ? bulletPoints.map(point => point.replace(/^[ ]*(?:[-*•]|\d+\.)[ ]+/, '').trim()).slice(0, 5)
        : ["Debug analysis based on your screenshots"];
      
      // Extract complexity information from debug content if available
      const timeComplexityPattern = /Time complexity:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*(?:Space complexity|$))/i;
      const spaceComplexityPattern = /Space complexity:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*(?:[A-Z]|$))/i;

      let debugTimeComplexity = "O(n) - Analysis based on debug feedback";
      let debugSpaceComplexity = "O(1) - Analysis based on debug feedback";

      const debugTimeMatch = debugContent.match(timeComplexityPattern);
      if (debugTimeMatch && debugTimeMatch[1]) {
        debugTimeComplexity = debugTimeMatch[1].trim();
        if (!debugTimeComplexity.match(/O\([^)]+\)/i)) {
          debugTimeComplexity = `O(n) - ${debugTimeComplexity}`;
        }
      }

      const debugSpaceMatch = debugContent.match(spaceComplexityPattern);
      if (debugSpaceMatch && debugSpaceMatch[1]) {
        debugSpaceComplexity = debugSpaceMatch[1].trim();
        if (!debugSpaceComplexity.match(/O\([^)]+\)/i)) {
          debugSpaceComplexity = `O(1) - ${debugSpaceComplexity}`;
        }
      }

      const response = {
        code: extractedCode,
        debug_analysis: formattedDebugContent,
        thoughts: thoughts,
        time_complexity: debugTimeComplexity,
        space_complexity: debugSpaceComplexity
      };

      return { success: true, data: response };
    } catch (error: any) {
      console.error("Debug processing error:", error);
      return { success: false, error: error.message || "Failed to process debug request" };
    }
  }

  private async processNonCodingScreenshots(
    screenshots: Array<{ path: string; data: string }>,
    signal: AbortSignal
  ) {
    try {
      const config = configHelper.loadConfig();
      const mainWindow = this.deps.getMainWindow();

      // Step 1: Analyze the question from screenshots
      const imageDataList = screenshots.map(screenshot => screenshot.data);

      // Update the user on progress
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Analyzing question from screenshots...",
          progress: 20
        });
      }

      let questionAnalysis;

      if (config.apiProvider === "openai") {
        if (!this.openaiClient) {
          return {
            success: false,
            error: "OpenAI API key not configured. Please check your settings."
          };
        }

        // Use OpenAI for processing
        const messages = [
          {
            role: "system" as const,
            content: "You are an expert assistant for online assessments. Carefully analyze the screenshot(s) to extract all questions (there may be 1-3 questions). Focus on capturing the EXACT questions as written, including all details, options, and requirements. If multiple questions exist, include all in the question_text field with clear numbering. Return the information in JSON format with these fields: question_text, question_type (mcq/fill_blank/behavioral/other), options (if applicable), context. Just return the structured JSON without any other text."
          },
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const,
                text: `Analyze this assessment question screenshot and extract all relevant information. The screenshot may contain 1-3 questions.

Return in JSON format with:
- question_text: The complete question text (if multiple questions, include all with clear numbering like "Q1: ... Q2: ... Q3: ...")
- question_type: Type of question (mcq, fill_blank, behavioral, other) - use the most common type if mixed
- options: Array of options if it's MCQ, null otherwise (for multiple MCQs, include all options)
- context: Any additional context or instructions`
              },
              ...imageDataList.map(data => ({
                type: "image_url" as const,
                image_url: { url: `data:image/png;base64,${data}` }
              }))
            ]
          }
        ];

        // Send to OpenAI Vision API
        const extractionResponse = await this.openaiClient.chat.completions.create({
          model: config.extractionModel || "gpt-4.1",
          messages: messages,
          max_tokens: 4000,
          temperature: 0.2
        });

        // Validate response structure
        if (!extractionResponse.choices || extractionResponse.choices.length === 0) {
          throw new Error("No response choices returned from OpenAI API");
        }

        const extractionContent = extractionResponse.choices[0].message.content;
        if (!extractionContent) {
          throw new Error("Empty response content from OpenAI API");
        }

        try {
          // Clean the response content by removing markdown code blocks if present
          let cleanedContent = extractionContent.trim();
          if (cleanedContent.startsWith('```json')) {
            cleanedContent = cleanedContent.replace(/^```json\s*/, '').replace(/\s*```$/, '');
          } else if (cleanedContent.startsWith('```')) {
            cleanedContent = cleanedContent.replace(/^```\s*/, '').replace(/\s*```$/, '');
          }

          questionAnalysis = JSON.parse(cleanedContent);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", extractionContent);
          throw new Error("Failed to parse question analysis from OpenAI");
        }
      } else if (config.apiProvider === "gemini") {
        if (!this.geminiApiKey) {
          return {
            success: false,
            error: "Gemini API key not configured. Please check your settings."
          };
        }

        try {
          const geminiMessages = [
            {
              role: "user",
              parts: [
                {
                  text: `You are an expert assistant for online assessments. Carefully analyze the screenshot(s) to extract the complete question. Focus on capturing the EXACT question as written, including all details, options, and requirements.

Return the information in JSON format with these fields: question_text, question_type (mcq/fill_blank/behavioral/other), options (if applicable), context. Just return the structured JSON without any other text.`
                },
                ...imageDataList.map(data => ({
                  inlineData: {
                    mimeType: "image/png",
                    data: data
                  }
                }))
              ]
            }
          ];

          // Make API request to Gemini
          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/${config.extractionModel || "gemini-2.5-flash"}:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 8000
              }
            },
            { signal }
          );

          const responseData = response.data;
          if (!responseData.candidates || responseData.candidates.length === 0) {
            throw new Error("No response from Gemini API");
          }

          const extractionContent = responseData.candidates[0].content.parts[0].text;
          try {
            // Clean the response content by removing markdown code blocks if present
            let cleanedContent = extractionContent.trim();
            if (cleanedContent.startsWith('```json')) {
              cleanedContent = cleanedContent.replace(/^```json\s*/, '').replace(/\s*```$/, '');
            } else if (cleanedContent.startsWith('```')) {
              cleanedContent = cleanedContent.replace(/^```\s*/, '').replace(/\s*```$/, '');
            }

            questionAnalysis = JSON.parse(cleanedContent);
          } catch (parseError) {
            console.error("Failed to parse Gemini response:", extractionContent);
            throw new Error("Failed to parse question analysis from Gemini");
          }
        } catch (error) {
          console.error("Error using Gemini API for question analysis:", error);
          return {
            success: false,
            error: "Failed to process with Gemini API. Please check your API key or try again later."
          };
        }
      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return {
            success: false,
            error: "Anthropic API key not configured. Please check your settings."
          };
        }

        try {
          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: `You are an expert assistant for online assessments. Carefully analyze the screenshot(s) to extract the complete question. Focus on capturing the EXACT question as written, including all details, options, and requirements.

Return the information in JSON format with these fields: question_text, question_type (mcq/fill_blank/behavioral/other), options (if applicable), context. Just return the structured JSON without any other text.`
                },
                ...imageDataList.map(data => ({
                  type: "image" as const,
                  source: {
                    type: "base64" as const,
                    media_type: "image/png" as const,
                    data: data
                  }
                }))
              ]
            }
          ];

          const response = await this.anthropicClient.messages.create({
            model: config.extractionModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages: messages,
            temperature: 0.2
          });

          // Validate response structure
          if (!response.content || response.content.length === 0) {
            throw new Error("No response content from Anthropic API");
          }

          const contentBlock = response.content[0] as { type: 'text', text: string };
          if (!contentBlock || contentBlock.type !== 'text' || !contentBlock.text) {
            throw new Error("Invalid response structure from Anthropic API");
          }

          const extractionContent = contentBlock.text;
          try {
            // Clean the response content by removing markdown code blocks if present
            let cleanedContent = extractionContent.trim();
            if (cleanedContent.startsWith('```json')) {
              cleanedContent = cleanedContent.replace(/^```json\s*/, '').replace(/\s*```$/, '');
            } else if (cleanedContent.startsWith('```')) {
              cleanedContent = cleanedContent.replace(/^```\s*/, '').replace(/\s*```$/, '');
            }

            questionAnalysis = JSON.parse(cleanedContent);
          } catch (parseError) {
            console.error("Failed to parse Anthropic response:", extractionContent);
            throw new Error("Failed to parse question analysis from Anthropic");
          }
        } catch (error: any) {
          console.error("Error using Anthropic API for question analysis:", error);
          return {
            success: false,
            error: "Failed to process with Anthropic API. Please check your API key or try again later."
          };
        }
      }

      // Update the user on progress
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Question analyzed successfully. Generating answer...",
          progress: 40
        });
      }

      // Store question info in AppState
      this.deps.setProblemInfo(questionAnalysis);

      // Send first success event
      if (mainWindow) {
        mainWindow.webContents.send(
          this.deps.PROCESSING_EVENTS.PROBLEM_EXTRACTED,
          questionAnalysis
        );

        // Generate answer after successful extraction
        const answerResult = await this.generateNonCodingAnswer(signal);
        if (answerResult.success) {
          // Final progress update
          mainWindow.webContents.send("processing-status", {
            message: "Answer generated successfully",
            progress: 100
          });

          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.SOLUTION_SUCCESS,
            answerResult.data
          );
          return { success: true, data: answerResult.data };
        } else {
          throw new Error(
            answerResult.error || "Failed to generate answer"
          );
        }
      }

      return { success: false, error: "No main window available" };
    } catch (error: any) {
      console.error("Non-coding processing error:", error);
      return { success: false, error: error.message || "Failed to process non-coding question" };
    }
  }

  private async generateNonCodingAnswer(signal: AbortSignal) {
    try {
      const questionInfo = this.deps.getProblemInfo();
      const config = configHelper.loadConfig();
      const mainWindow = this.deps.getMainWindow();

      if (!questionInfo) {
        throw new Error("No question info available");
      }

      // Update progress status
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Analyzing question and generating answer...",
          progress: 60
        });
      }

      // Create prompt for answer generation based on question type
      let promptText = "";

      if (questionInfo.question_type === "mcq") {
        promptText = `
Analyze the multiple choice question(s) from the screenshot and provide concise answers:

Question(s): ${questionInfo.question_text}

${questionInfo.options ? `Options:
${questionInfo.options.map((opt: string, idx: number) => `${String.fromCharCode(65 + idx)}. ${opt}`).join('\n')}` : ''}

${questionInfo.context ? `Context: ${questionInfo.context}` : ''}

For each question, provide:
- **Answer:** [Letter] - [Option text]
- **Reason:** [Brief 1-2 sentence explanation]

If there are multiple questions, number them clearly (Q1, Q2, Q3). Keep explanations concise and focused.
`;
      } else if (questionInfo.question_type === "fill_blank") {
        promptText = `
Analyze the fill-in-the-blank question(s) from the screenshot and provide concise answers:

Question(s): ${questionInfo.question_text}

${questionInfo.context ? `Context: ${questionInfo.context}` : ''}

For each question, provide:
- **Answer:** [Word/phrase to fill the blank]
- **Reason:** [Brief 1-2 sentence explanation]

If there are multiple questions, number them clearly (Q1, Q2, Q3). Keep explanations concise and focused.
`;
      } else if (questionInfo.question_type === "behavioral") {
        promptText = `
Analyze this behavioral/situational question and provide a structured response:

Question: ${questionInfo.question_text}

${questionInfo.context ? `Context: ${questionInfo.context}` : ''}

Structure your response as follows:

1. **Tips** or **How to answer this:** section with 3-4 concise bullet points on how to approach this question effectively

2. **Sample Answer:** A natural, conversational first-person response as if you're speaking directly to an interviewer. Use the STAR method (Situation, Task, Action, Result) structure but write it in a flowing, first-person narrative without explicit subheadings. The response should:
   - Be written in first person ("I", "my", "we")
   - Sound natural and conversational, like you're telling a story
   - Be sufficiently detailed but not overly long (aim for 2-3 paragraphs)
   - Flow naturally from situation to task to action to result
   - Show specific examples and quantifiable results where possible
   - Demonstrate relevant skills and qualities

Format with clear headings and write the sample answer as if you're actually answering the question in an interview setting.
`;
      } else {
        promptText = `
Analyze the assessment question(s) from the screenshot and provide comprehensive answers:

Question(s): ${questionInfo.question_text}

${questionInfo.context ? `Context: ${questionInfo.context}` : ''}

For each question, provide:
- **Answer:** [Clear, direct response]
- **Explanation:** [Supporting reasoning and relevant details]

If there are multiple questions, number them clearly (Q1, Q2, Q3). Be thorough but organized in your responses.
`;
      }

      let responseContent;

      if (config.apiProvider === "openai") {
        // OpenAI processing
        if (!this.openaiClient) {
          return {
            success: false,
            error: "OpenAI API key not configured. Please check your settings."
          };
        }

        // Send to OpenAI API
        const systemPrompt = questionInfo.question_type === "behavioral"
          ? "You are an expert interview coach. For behavioral questions, provide a structured response with: 1) A 'Tips' or 'How to answer this:' section with 3-4 concise bullet points, followed by 2) A 'Sample Answer:' section with a natural, first-person response using STAR method naturally. Be conversational, specific, and authentic."
          : questionInfo.question_type === "mcq" || questionInfo.question_type === "fill_blank"
          ? "You are an expert assessment assistant. For MCQs and fill-in-the-blank questions, provide concise answers with brief explanations. Handle multiple questions clearly and keep responses focused and direct."
          : "You are an expert assessment assistant. Provide clear, comprehensive answers with organized explanations. Structure your responses clearly for easy understanding.";

        const answerResponse = await this.openaiClient.chat.completions.create({
          model: config.solutionModel || "gpt-4.1",
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: promptText }
          ],
          max_tokens: 4000,
          temperature: 0.2
        });

        // Validate response structure
        if (!answerResponse.choices || answerResponse.choices.length === 0) {
          throw new Error("No response choices returned from OpenAI API");
        }

        responseContent = answerResponse.choices[0].message.content;

        if (!responseContent) {
          throw new Error("Empty response content from OpenAI API");
        }
      } else if (config.apiProvider === "gemini") {
        if (!this.geminiApiKey) {
          return {
            success: false,
            error: "Gemini API key not configured. Please check your settings."
          };
        }

        try {
          // Create Gemini message structure
          const systemPrompt = questionInfo.question_type === "behavioral"
            ? "You are an expert interview coach. For behavioral questions, provide a structured response with: 1) A 'Tips' or 'How to answer this:' section with 3-4 concise bullet points, followed by 2) A 'Sample Answer:' section with a natural, first-person response using STAR method naturally. Be conversational, specific, and authentic."
            : questionInfo.question_type === "mcq" || questionInfo.question_type === "fill_blank"
            ? "You are an expert assessment assistant. For MCQs and fill-in-the-blank questions, provide concise answers with brief explanations. Handle multiple questions clearly and keep responses focused and direct."
            : "You are an expert assessment assistant. Provide clear, comprehensive answers with organized explanations. Structure your responses clearly for easy understanding.";

          const geminiMessages = [
            {
              role: "user",
              parts: [
                {
                  text: `${systemPrompt}\n\n${promptText}`
                }
              ]
            }
          ];

          // Make API request to Gemini
          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/${config.solutionModel || "gemini-2.5-flash"}:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 8000
              }
            },
            { signal }
          );

          const responseData = response.data;
          if (!responseData.candidates || responseData.candidates.length === 0) {
            throw new Error("No response from Gemini API");
          }

          const candidate = responseData.candidates[0];
          if (!candidate.content || !candidate.content.parts || candidate.content.parts.length === 0) {
            throw new Error("Invalid response structure from Gemini API");
          }

          responseContent = candidate.content.parts[0].text;

          if (!responseContent) {
            throw new Error("Empty response content from Gemini API");
          }
        } catch (error) {
          console.error("Error using Gemini API for answer generation:", error);
          return {
            success: false,
            error: "Failed to generate answer with Gemini API. Please check your API key or try again later."
          };
        }
      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return {
            success: false,
            error: "Anthropic API key not configured. Please check your settings."
          };
        }

        try {
          const systemPrompt = questionInfo.question_type === "behavioral"
            ? "You are an expert interview coach. For behavioral questions, provide a structured response with: 1) A 'Tips' or 'How to answer this:' section with 3-4 concise bullet points, followed by 2) A 'Sample Answer:' section with a natural, first-person response using STAR method naturally. Be conversational, specific, and authentic."
            : questionInfo.question_type === "mcq" || questionInfo.question_type === "fill_blank"
            ? "You are an expert assessment assistant. For MCQs and fill-in-the-blank questions, provide concise answers with brief explanations. Handle multiple questions clearly and keep responses focused and direct."
            : "You are an expert assessment assistant. Provide clear, comprehensive answers with organized explanations. Structure your responses clearly for easy understanding.";

          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: `${systemPrompt}\n\n${promptText}`
                }
              ]
            }
          ];

          // Send to Anthropic API
          const response = await this.anthropicClient.messages.create({
            model: config.solutionModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages: messages,
            temperature: 0.2
          });

          // Validate response structure
          if (!response.content || response.content.length === 0) {
            throw new Error("No response content from Anthropic API");
          }

          const contentBlock = response.content[0] as { type: 'text', text: string };
          if (!contentBlock || contentBlock.type !== 'text' || !contentBlock.text) {
            throw new Error("Invalid response structure from Anthropic API");
          }

          responseContent = contentBlock.text;
        } catch (error: any) {
          console.error("Error using Anthropic API for answer generation:", error);
          return {
            success: false,
            error: "Failed to generate answer with Anthropic API. Please check your API key or try again later."
          };
        }
      }

      if (!responseContent) {
        throw new Error("No response content generated");
      }

      const formattedResponse = {
        answer: responseContent,
        question_type: questionInfo.question_type,
        question_text: questionInfo.question_text
      };

      return { success: true, data: formattedResponse };
    } catch (error: any) {
      console.error("Answer generation error:", error);
      return { success: false, error: error.message || "Failed to generate answer" };
    }
  }

  public cancelOngoingRequests(): void {
    let wasCancelled = false

    if (this.currentProcessingAbortController) {
      this.currentProcessingAbortController.abort()
      this.currentProcessingAbortController = null
      wasCancelled = true
    }

    if (this.currentExtraProcessingAbortController) {
      this.currentExtraProcessingAbortController.abort()
      this.currentExtraProcessingAbortController = null
      wasCancelled = true
    }

    this.deps.setHasDebugged(false)

    this.deps.setProblemInfo(null)

    const mainWindow = this.deps.getMainWindow()
    if (wasCancelled && mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS)
    }
  }
}
