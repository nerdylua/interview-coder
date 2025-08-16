// ScreenshotHelper.ts

import path from "node:path";
import fs from "node:fs";
import { app } from "electron";
import { v4 as uuidv4 } from "uuid";
import { execFile } from "child_process";
import { promisify } from "util";
import screenshot from "screenshot-desktop";
import os from "os";

const execFileAsync = promisify(execFile);

export type ScreenshotMode = "full" | "left" | "right";

export class ScreenshotHelper {
  private screenshotQueue: string[] = [];
  private extraScreenshotQueue: string[] = [];
  private readonly MAX_SCREENSHOTS = 5;
  private readonly HORIZONTAL_EXTENSION = 387;
  private readonly VERTICAL_EXTENSION = 136; // Extra pixels to capture at the bottom

  private readonly screenshotDir: string;
  private readonly extraScreenshotDir: string;
  private readonly tempDir: string;

  private view: "queue" | "solutions" | "debug" = "queue";
  private screenshotMode: ScreenshotMode = "full";

  constructor(view: "queue" | "solutions" | "debug" = "queue") {
    this.view = view;

    // Initialize directories
    this.screenshotDir = path.join(app.getPath("userData"), "screenshots");
    this.extraScreenshotDir = path.join(
      app.getPath("userData"),
      "extra_screenshots"
    );
    this.tempDir = path.join(
      app.getPath("temp"),
      "interview-coder-screenshots"
    );

    // Create directories if they don't exist
    this.ensureDirectoriesExist();

    // Clean existing screenshot directories when starting the app
    this.cleanScreenshotDirectories();
  }

  private ensureDirectoriesExist(): void {
    const directories = [
      this.screenshotDir,
      this.extraScreenshotDir,
      this.tempDir,
    ];

    for (const dir of directories) {
      if (!fs.existsSync(dir)) {
        try {
          fs.mkdirSync(dir, { recursive: true });
          console.log(`Created directory: ${dir}`);
        } catch (err) {
          console.error(`Error creating directory ${dir}:`, err);
        }
      }
    }
  }

  // This method replaces loadExistingScreenshots() to ensure we start with empty queues
  private cleanScreenshotDirectories(): void {
    try {
      // Clean main screenshots directory
      if (fs.existsSync(this.screenshotDir)) {
        const files = fs
          .readdirSync(this.screenshotDir)
          .filter((file) => file.endsWith(".png"))
          .map((file) => path.join(this.screenshotDir, file));

        // Delete each screenshot file
        for (const file of files) {
          try {
            fs.unlinkSync(file);
            console.log(`Deleted existing screenshot: ${file}`);
          } catch (err) {
            console.error(`Error deleting screenshot ${file}:`, err);
          }
        }
      }

      // Clean extra screenshots directory
      if (fs.existsSync(this.extraScreenshotDir)) {
        const files = fs
          .readdirSync(this.extraScreenshotDir)
          .filter((file) => file.endsWith(".png"))
          .map((file) => path.join(this.extraScreenshotDir, file));

        // Delete each screenshot file
        for (const file of files) {
          try {
            fs.unlinkSync(file);
            console.log(`Deleted existing extra screenshot: ${file}`);
          } catch (err) {
            console.error(`Error deleting extra screenshot ${file}:`, err);
          }
        }
      }

      console.log("Screenshot directories cleaned successfully");
    } catch (err) {
      console.error("Error cleaning screenshot directories:", err);
    }
  }

  public getView(): "queue" | "solutions" | "debug" {
    return this.view;
  }

  public setView(view: "queue" | "solutions" | "debug"): void {
    console.log("Setting view in ScreenshotHelper:", view);
    console.log(
      "Current queues - Main:",
      this.screenshotQueue,
      "Extra:",
      this.extraScreenshotQueue
    );
    this.view = view;
  }

  public getScreenshotMode(): ScreenshotMode {
    return this.screenshotMode;
  }

  public setScreenshotMode(mode: ScreenshotMode): void {
    console.log("Setting screenshot mode:", mode);
    this.screenshotMode = mode;
  }

  public getScreenshotQueue(): string[] {
    return this.screenshotQueue;
  }

  public getExtraScreenshotQueue(): string[] {
    console.log("Getting extra screenshot queue:", this.extraScreenshotQueue);
    return this.extraScreenshotQueue;
  }

  public clearQueues(): void {
    // Clear screenshotQueue
    this.screenshotQueue.forEach((screenshotPath) => {
      fs.unlink(screenshotPath, (err) => {
        if (err)
          console.error(`Error deleting screenshot at ${screenshotPath}:`, err);
      });
    });
    this.screenshotQueue = [];

    // Clear extraScreenshotQueue
    this.extraScreenshotQueue.forEach((screenshotPath) => {
      fs.unlink(screenshotPath, (err) => {
        if (err)
          console.error(
            `Error deleting extra screenshot at ${screenshotPath}:`,
            err
          );
      });
    });
    this.extraScreenshotQueue = [];
  }

  private async captureScreenshot(): Promise<Buffer> {
    try {
      console.log(`Starting screenshot capture in ${this.screenshotMode} mode...`);

      // For Windows, try multiple methods
      if (process.platform === "win32") {
        return await this.captureWindowsScreenshot();
      }

      // For macOS and Linux, use buffer directly
      console.log("Taking screenshot on non-Windows platform");
      const buffer = await screenshot({ format: "png" });
      console.log(
        `Screenshot captured successfully, size: ${buffer.length} bytes`
      );

      // Apply cropping based on mode
      return await this.applyCropping(buffer);
    } catch (error) {
      console.error("Error capturing screenshot:", error);
      throw new Error(`Failed to capture screenshot: ${error.message}`);
    }
  }



  /**
   * Apply cropping to screenshot based on current mode
   */
  private async applyCropping(buffer: Buffer): Promise<Buffer> {
    if (this.screenshotMode === "full") {
      return buffer;
    }

    try {
      // We'll use sharp for image processing if available, otherwise fall back to basic cropping
      const sharp = require('sharp');
      const image = sharp(buffer);
      const metadata = await image.metadata();

      if (!metadata.width || !metadata.height) {
        console.warn("Could not get image dimensions, returning full screenshot");
        return buffer;
      }

      const width = metadata.width;
      const height = metadata.height;

      console.log(`Original screenshot dimensions: ${width}x${height}`);

      // Account for the extended width we added when calculating crops
      const extendedWidth = this.HORIZONTAL_EXTENSION;

      let cropOptions;
      if (this.screenshotMode === "left") {
        // For left half, use half of the effective width
        const leftWidth = Math.floor((width - extendedWidth) / 2) + Math.floor(extendedWidth / 2);
        cropOptions = {
          left: 0,
          top: 0,
          width: leftWidth,
          height: height
        };
      } else if (this.screenshotMode === "right") {
        // For right half, start from the middle and go to the end (including extended area)
        const leftWidth = Math.floor((width - extendedWidth) / 2) + Math.floor(extendedWidth / 2);
        cropOptions = {
          left: leftWidth,
          top: 0,
          width: width - leftWidth, // This captures the entire right half including extension
          height: height
        };
      }

      if (cropOptions) {
        console.log(`Applying ${this.screenshotMode} cropping:`, cropOptions);
        const croppedBuffer = await image.extract(cropOptions).png().toBuffer();
        console.log(`Applied ${this.screenshotMode} cropping: ${cropOptions.width}x${cropOptions.height} at position (${cropOptions.left}, ${cropOptions.top})`);
        return croppedBuffer;
      }

      return buffer;
    } catch (error) {
      console.warn("Failed to apply cropping, falling back to full screenshot:", error);
      return buffer;
    }
  }

  /**
   * Windows-specific screenshot capture using PowerShell
   */
  private async captureWindowsScreenshot(): Promise<Buffer> {
    console.log("Taking Windows screenshot with PowerShell");

    try {
      const tempFile = path.join(this.tempDir, `ps-temp-${uuidv4()}.png`);

      // PowerShell command to take screenshot using .NET classes with extensions
      const psScript = `
      Add-Type -AssemblyName System.Windows.Forms,System.Drawing
      try {
        $screens = [System.Windows.Forms.Screen]::AllScreens
        Write-Host "Found $($screens.Count) screens"

        # Get the virtual screen bounds (covers all monitors)
        $left = [System.Windows.Forms.SystemInformation]::VirtualScreen.Left
        $top = [System.Windows.Forms.SystemInformation]::VirtualScreen.Top
        $width = [System.Windows.Forms.SystemInformation]::VirtualScreen.Width
        $height = [System.Windows.Forms.SystemInformation]::VirtualScreen.Height

        # Add extra dimensions to capture missing areas
        $extraWidth = ${this.HORIZONTAL_EXTENSION}
        $extraHeight = ${this.VERTICAL_EXTENSION}
        $width = $width + $extraWidth
        $height = $height + $extraHeight

        Write-Host "Virtual screen (extended): Left=$left, Top=$top, Width=$width, Height=$height"

        if ($width -le 0 -or $height -le 0) {
          throw "Invalid screen dimensions: width=$width, height=$height"
        }

        $bmp = New-Object System.Drawing.Bitmap $width, $height
        $graphics = [System.Drawing.Graphics]::FromImage($bmp)
        $graphics.CopyFromScreen($left, $top, 0, 0, [System.Drawing.Size]::new($width, $height))
        $bmp.Save('${tempFile.replace(
          /\\/g,
          "\\\\"
        )}', [System.Drawing.Imaging.ImageFormat]::Png)
        $graphics.Dispose()
        $bmp.Dispose()
        Write-Host "Screenshot saved successfully"
      } catch {
        Write-Error "PowerShell screenshot failed: $_"
        exit 1
      }
      `;

      // Execute PowerShell
      await execFileAsync("powershell", [
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        psScript,
      ]);

      // Check if file exists and read it
      if (fs.existsSync(tempFile)) {
        const buffer = await fs.promises.readFile(tempFile);
        console.log(
          `PowerShell screenshot successful, size: ${buffer.length} bytes`
        );

        // Cleanup
        try {
          await fs.promises.unlink(tempFile);
        } catch (err) {
          console.warn("Failed to clean up PowerShell temp file:", err);
        }

        return await this.applyCropping(buffer);
      } else {
        throw new Error("PowerShell screenshot file not created");
      }
    } catch (error) {
      console.error("PowerShell screenshot failed:", error);
      throw new Error(
        "Could not capture screenshot. Please check your Windows security settings and try again."
      );
    }
  }



  public async takeScreenshot(
    hideMainWindow: () => void,
    showMainWindow: () => void
  ): Promise<string> {
    console.log("Taking screenshot in view:", this.view);
    hideMainWindow();

    // Increased delay for window hiding on Windows
    const hideDelay = process.platform === "win32" ? 500 : 300;
    await new Promise((resolve) => setTimeout(resolve, hideDelay));

    let screenshotPath = "";
    try {
      // Get screenshot buffer using cross-platform method
      const screenshotBuffer = await this.captureScreenshot();

      if (!screenshotBuffer || screenshotBuffer.length === 0) {
        throw new Error("Screenshot capture returned empty buffer");
      }

      // Save and manage the screenshot based on current view
      if (this.view === "queue") {
        screenshotPath = path.join(this.screenshotDir, `${uuidv4()}.png`);
        const screenshotDir = path.dirname(screenshotPath);
        if (!fs.existsSync(screenshotDir)) {
          fs.mkdirSync(screenshotDir, { recursive: true });
        }
        await fs.promises.writeFile(screenshotPath, screenshotBuffer);
        console.log("Adding screenshot to main queue:", screenshotPath);
        this.screenshotQueue.push(screenshotPath);
        if (this.screenshotQueue.length > this.MAX_SCREENSHOTS) {
          const removedPath = this.screenshotQueue.shift();
          if (removedPath) {
            try {
              await fs.promises.unlink(removedPath);
              console.log(
                "Removed old screenshot from main queue:",
                removedPath
              );
            } catch (error) {
              console.error("Error removing old screenshot:", error);
            }
          }
        }
      } else {
        // In solutions view, only add to extra queue
        screenshotPath = path.join(this.extraScreenshotDir, `${uuidv4()}.png`);
        const screenshotDir = path.dirname(screenshotPath);
        if (!fs.existsSync(screenshotDir)) {
          fs.mkdirSync(screenshotDir, { recursive: true });
        }
        await fs.promises.writeFile(screenshotPath, screenshotBuffer);
        console.log("Adding screenshot to extra queue:", screenshotPath);
        this.extraScreenshotQueue.push(screenshotPath);
        if (this.extraScreenshotQueue.length > this.MAX_SCREENSHOTS) {
          const removedPath = this.extraScreenshotQueue.shift();
          if (removedPath) {
            try {
              await fs.promises.unlink(removedPath);
              console.log(
                "Removed old screenshot from extra queue:",
                removedPath
              );
            } catch (error) {
              console.error("Error removing old screenshot:", error);
            }
          }
        }
      }
    } catch (error) {
      console.error("Screenshot error:", error);
      throw error;
    } finally {
      // Increased delay for showing window again
      await new Promise((resolve) => setTimeout(resolve, 200));
      showMainWindow();
    }

    return screenshotPath;
  }

  public async getImagePreview(filepath: string): Promise<string> {
    try {
      if (!fs.existsSync(filepath)) {
        console.error(`Image file not found: ${filepath}`);
        return "";
      }

      const data = await fs.promises.readFile(filepath);
      return `data:image/png;base64,${data.toString("base64")}`;
    } catch (error) {
      console.error("Error reading image:", error);
      return "";
    }
  }

  public async deleteScreenshot(
    path: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      if (fs.existsSync(path)) {
        await fs.promises.unlink(path);
      }

      if (this.view === "queue") {
        this.screenshotQueue = this.screenshotQueue.filter(
          (filePath) => filePath !== path
        );
      } else {
        this.extraScreenshotQueue = this.extraScreenshotQueue.filter(
          (filePath) => filePath !== path
        );
      }
      return { success: true };
    } catch (error) {
      console.error("Error deleting file:", error);
      return { success: false, error: error.message };
    }
  }

  public clearExtraScreenshotQueue(): void {
    // Clear extraScreenshotQueue
    this.extraScreenshotQueue.forEach((screenshotPath) => {
      if (fs.existsSync(screenshotPath)) {
        fs.unlink(screenshotPath, (err) => {
          if (err)
            console.error(
              `Error deleting extra screenshot at ${screenshotPath}:`,
              err
            );
        });
      }
    });
    this.extraScreenshotQueue = [];
  }
}
