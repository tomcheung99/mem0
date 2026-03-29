import { useEffect, useState } from "react";
import { useMemoriesApi } from "@/hooks/useMemoriesApi";
import { useSelector } from "react-redux";
import { RootState } from "@/store/store";
import { VersionHistoryEntry } from "@/store/memoriesSlice";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  GitCommitHorizontal,
  Merge,
  Copy,
  RotateCcw,
  PlusCircle,
} from "lucide-react";

const CHANGE_TYPE_CONFIG: Record<
  string,
  { label: string; icon: React.ReactNode; color: string }
> = {
  update: {
    label: "Updated",
    icon: <GitCommitHorizontal size={14} />,
    color: "text-blue-400",
  },
  add_overwrite: {
    label: "Overwritten on add",
    icon: <PlusCircle size={14} />,
    color: "text-green-400",
  },
  merge: {
    label: "Merged",
    icon: <Merge size={14} />,
    color: "text-purple-400",
  },
  dedup: {
    label: "Deduplicated",
    icon: <Copy size={14} />,
    color: "text-yellow-400",
  },
  restore: {
    label: "Restored",
    icon: <RotateCcw size={14} />,
    color: "text-orange-400",
  },
};

function getConfig(changeType: string) {
  return (
    CHANGE_TYPE_CONFIG[changeType] ?? {
      label: changeType,
      icon: <GitCommitHorizontal size={14} />,
      color: "text-zinc-400",
    }
  );
}

interface VersionHistoryProps {
  memoryId: string;
}

export function VersionHistory({ memoryId }: VersionHistoryProps) {
  const { fetchVersionHistory } = useMemoriesApi();
  const versions = useSelector(
    (state: RootState) => state.memories.versionHistory
  );
  const [isLoading, setIsLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        await fetchVersionHistory(memoryId);
      } catch (error) {
        console.error("Failed to fetch version history:", error);
      } finally {
        setIsLoading(false);
      }
    };
    load();
  }, []);

  if (isLoading) {
    return (
      <div className="w-full max-w-md mx-auto rounded-lg overflow-hidden bg-zinc-900 border border-zinc-800 text-white p-6">
        <p className="text-center text-zinc-500">Loading version history...</p>
      </div>
    );
  }

  return (
    <div className="w-full max-w-md mx-auto rounded-lg overflow-hidden bg-zinc-900 border border-zinc-800 text-white pb-1">
      <div className="px-6 py-4 flex justify-between items-center bg-zinc-800 border-b border-zinc-800">
        <h2 className="font-semibold">Version History</h2>
        <span className="text-xs text-zinc-500">
          {versions.length} version{versions.length !== 1 ? "s" : ""}
        </span>
      </div>

      <ScrollArea className="p-6 max-h-[450px]">
        {versions.length === 0 && (
          <div className="w-full max-w-md mx-auto rounded-3xl overflow-hidden min-h-[110px] flex items-center justify-center text-white p-6">
            <p className="text-center text-zinc-500">
              No version history available
            </p>
          </div>
        )}
        <ul className="space-y-6">
          {versions.map((entry: VersionHistoryEntry, index: number) => {
            const config = getConfig(entry.change_type);
            const isExpanded = expandedId === entry.id;

            return (
              <li key={entry.id} className="relative flex items-start gap-4">
                {/* Icon */}
                <div
                  className={`relative z-10 rounded-full bg-zinc-800 w-8 h-8 flex items-center justify-center flex-shrink-0 ${config.color}`}
                >
                  {config.icon}
                </div>

                {/* Connector line */}
                {index < versions.length - 1 && (
                  <div className="absolute left-4 top-6 bottom-0 w-[1px] h-[calc(100%+0.5rem)] bg-zinc-700 transform -translate-x-1/2" />
                )}

                {/* Content */}
                <div className="flex flex-col gap-1 min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-medium ${config.color}`}>
                      v{entry.version}
                    </span>
                    <span className="text-xs text-zinc-500">
                      {config.label}
                    </span>
                  </div>

                  {entry.created_at && (
                    <span className="text-zinc-500 text-xs">
                      {new Date(entry.created_at + "Z").toLocaleDateString(
                        "en-US",
                        {
                          year: "numeric",
                          month: "short",
                          day: "numeric",
                          hour: "numeric",
                          minute: "numeric",
                        }
                      )}
                    </span>
                  )}

                  <button
                    className="text-xs text-zinc-400 hover:text-white mt-1 text-left w-fit"
                    onClick={() =>
                      setExpandedId(isExpanded ? null : entry.id)
                    }
                  >
                    {isExpanded ? "Hide diff" : "Show diff"}
                  </button>

                  {isExpanded && (
                    <div className="mt-2 space-y-2 text-xs">
                      {entry.old_content && (
                        <div className="rounded bg-red-950/30 border border-red-900/40 p-2">
                          <span className="text-red-400 font-mono">−</span>{" "}
                          <span className="text-red-300 break-words">
                            {entry.old_content}
                          </span>
                        </div>
                      )}
                      <div className="rounded bg-green-950/30 border border-green-900/40 p-2">
                        <span className="text-green-400 font-mono">+</span>{" "}
                        <span className="text-green-300 break-words">
                          {entry.new_content}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      </ScrollArea>
    </div>
  );
}
