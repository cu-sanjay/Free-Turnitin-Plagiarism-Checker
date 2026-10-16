import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClientProvider } from "@tanstack/react-query";
import { Route, Switch } from "wouter";
import { queryClient } from "@/lib/queryClient";
import AppShell from "@/components/AppShell";
import Index from "./pages/Index";
import HistoryPage from "./pages/HistoryPage";
import ToolsPage from "./pages/ToolsPage";
import AboutPage from "./pages/AboutPage";
import NotFound from "./pages/NotFound";

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <AppShell>
        <Switch>
          <Route path="/" component={Index} />
          <Route path="/history" component={HistoryPage} />
          <Route path="/tools" component={ToolsPage} />
          <Route path="/about" component={AboutPage} />
          <Route component={NotFound} />
        </Switch>
      </AppShell>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
