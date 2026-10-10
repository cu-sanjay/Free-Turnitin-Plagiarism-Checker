import { app, BrowserWindow, shell } from "electron";
import path from "node:path";
import { pathToFileURL } from "node:url";

let mainWindow;
let server;

async function waitForServer(url) {
  for (let attempt = 0; attempt < 50; attempt++) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  throw new Error("ClearMark could not start its local app server.");
}

async function createWindow() {
  process.env.NODE_ENV = "production";
  const { startServer } = await import("../server/index.js");
  const started = await startServer({ port: 0, host: "127.0.0.1" });
  server = started.server;
  const appUrl = `http://127.0.0.1:${started.port}/`;
  await waitForServer(appUrl);

  mainWindow = new BrowserWindow({
    width: 1180,
    height: 860,
    minWidth: 760,
    minHeight: 640,
    show: false,
    backgroundColor: "#f6f8f6",
    title: "ClearMark",
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://")) void shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (url !== appUrl) event.preventDefault();
  });
  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  await mainWindow.loadURL(appUrl);
}

app.whenReady().then(createWindow).catch((error) => {
  console.error("Could not launch ClearMark:", error);
  app.quit();
});

app.on("window-all-closed", () => {
  if (server) server.close();
  app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow().catch((error) => {
      console.error("Could not reopen ClearMark:", error);
      app.quit();
    });
  }
});
