import { Action, ActionPanel, Grid, Icon } from "@raycast/api";
import { request as httpRequest } from "node:http";
import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8000/search";
const RESULT_LIMIT = 12;
const DEBOUNCE_MS = 250;

type SearchHit = {
  path: string;
  modality: "text" | "image" | "pdf";
  score: number;
  preview?: string;
  thumbnail_path?: string;
  metadata: {
    filename?: string;
  };
};

type SearchResponse = {
  query: string;
  hits: SearchHit[];
};

export default function SearchCommand() {
  const [searchText, setSearchText] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>();
  const debouncedQuery = useDebouncedValue(searchText.trim(), DEBOUNCE_MS);

  useEffect(() => {
    const controller = new AbortController();

    if (!debouncedQuery) {
      setHits([]);
      setError(undefined);
      setIsLoading(false);
      return () => controller.abort();
    }

    setIsLoading(true);
    setError(undefined);

    void postSearchRequest(debouncedQuery, controller.signal)
      .then((payload) => {
        setHits(payload.hits || []);
      })
      .catch((requestError: Error) => {
        if (controller.signal.aborted || requestError.message === "aborted") {
          return;
        }

        setHits([]);
        setError(
          requestError.message === "Failed to fetch"
            ? "The local API is not reachable. Start FastAPI on http://127.0.0.1:8000."
            : requestError.message,
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      });

    return () => controller.abort();
  }, [debouncedQuery]);

  return (
    <Grid
      isLoading={isLoading}
      onSearchTextChange={setSearchText}
      searchBarPlaceholder="Search your local memory..."
      columns={4}
      aspectRatio="4/3"
      fit={Grid.Fit.Fill}
      inset={Grid.Inset.Small}
      throttle
    >
      {hits.map((hit) => (
        <Grid.Item
          key={hit.path}
          content={hit.thumbnail_path || { fileIcon: hit.path }}
          title={hit.metadata.filename || fileNameFromPath(hit.path)}
          subtitle={`${hit.modality.toUpperCase()}  ${hit.score.toFixed(3)}`}
          accessory={{ icon: iconForModality(hit.modality), tooltip: shortenPreview(hit.preview) }}
          quickLook={{ path: hit.path }}
          keywords={buildKeywords(hit)}
          actions={
            <ActionPanel>
              <Action.Open title="Open File" target={hit.path} />
              <Action.ShowInFinder title="Reveal in Finder" path={hit.path} />
              <Action.ToggleQuickLook />
              <Action.CopyToClipboard title="Copy Full Path" content={hit.path} />
            </ActionPanel>
          }
        />
      ))}
      <Grid.EmptyView
        icon={error ? Icon.Warning : Icon.MagnifyingGlass}
        title={emptyStateTitle(debouncedQuery, error)}
        description={emptyStateDescription(debouncedQuery, error)}
      />
    </Grid>
  );
}

function useDebouncedValue(value: string, delayMs: number): string {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);

  return debouncedValue;
}

function emptyStateTitle(query: string, error?: string): string {
  if (error) {
    return "Local API Unavailable";
  }
  if (!query) {
    return "Search Your Local Memory";
  }
  return "No Results";
}

function emptyStateDescription(query: string, error?: string): string {
  if (error) {
    return error;
  }
  if (!query) {
    return "Type a natural-language query to search text files, PDFs, and images.";
  }
  return "Try a broader phrase or re-run indexing after changing the dataset.";
}

function fileNameFromPath(path: string): string {
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

function shortenPreview(preview?: string): string {
  if (!preview) {
    return "No preview available";
  }
  return preview.length > 160 ? `${preview.slice(0, 157)}...` : preview;
}

function iconForModality(modality: SearchHit["modality"]): Icon {
  switch (modality) {
    case "image":
      return Icon.Image;
    case "pdf":
      return Icon.Document;
    default:
      return Icon.TextDocument;
  }
}

function buildKeywords(hit: SearchHit): string[] {
  return [hit.metadata.filename, hit.preview, hit.modality].filter(Boolean) as string[];
}

function postSearchRequest(query: string, signal: AbortSignal): Promise<SearchResponse> {
  const body = JSON.stringify({ query, k: RESULT_LIMIT });
  const url = new URL(API_URL);

  return new Promise((resolve, reject) => {
    const req = httpRequest(
      {
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port,
        path: url.pathname,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
        },
      },
      (res) => {
        let responseBody = "";

        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          responseBody += chunk;
        });
        res.on("end", () => {
          try {
            const payload = JSON.parse(responseBody || "{}") as SearchResponse | { detail?: string };
            const statusCode = res.statusCode ?? 500;

            if (statusCode < 200 || statusCode >= 300) {
              const detail = "detail" in payload ? payload.detail : undefined;
              reject(new Error(detail || `Search failed with status ${statusCode}`));
              return;
            }

            resolve(payload as SearchResponse);
          } catch (error) {
            reject(error instanceof Error ? error : new Error("Failed to parse API response."));
          }
        });
      },
    );

    req.on("error", (error) => {
      reject(error);
    });

    signal.addEventListener("abort", () => {
      req.destroy();
      reject(new Error("aborted"));
    });

    req.write(body);
    req.end();
  });
}
