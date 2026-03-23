import { memo } from "react";
import { EmptyState } from "./ui.jsx";
import { formatTimestamp, formatRelativeTime, prettyLabel } from "../utils.js";

function stripMd(text) {
  return String(text || "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/_([^_]+)_/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
}

export default memo(function RedditIntelRail({ feed, onImportPost, pendingKey }) {
  const posts = Array.isArray(feed?.posts) ? feed.posts : [];
  const fetchedAt = feed?.fetched_at ? formatTimestamp(feed.fetched_at) : "–";

  return (
    <aside className="workspace-rail workspace-rail-right">
      <article className="panel reddit-rail-card">
        <div className="reddit-rail-head">
          <div>
            <p className="eyebrow">External watch</p>
            <h3 className="reddit-rail-title">r/Collatz feed</h3>
          </div>
          <a
            className="reddit-open-reddit-link"
            href="https://www.reddit.com/r/Collatz/"
            target="_blank"
            rel="noreferrer"
          >
            Open ↗
          </a>
        </div>
        <p className="reddit-rail-note">
          Intake only — nothing here is trusted automatically.
        </p>
        <div className="reddit-rail-meta-row">
          <span className="reddit-meta-chip">fetched {fetchedAt}</span>
          {feed?.review_candidate_count > 0 && (
            <span className="reddit-meta-chip reddit-meta-chip-alert">
              {feed.review_candidate_count} for review
            </span>
          )}
        </div>
        {posts.length === 0 ? (
          <EmptyState
            title="No subreddit feed yet"
            text="Backend hasn't returned r/Collatz posts yet."
          />
        ) : (
          <div className="reddit-feed-list">
            {posts.map((post) => {
              const isOwn = post.is_own === true;
              return (
                <article
                  key={post.id}
                  className={`reddit-post-card${isOwn ? " reddit-post-card-own" : ""}`}
                >
                  <div className="reddit-post-header">
                    <div className="reddit-post-header-left">
                      {isOwn && <span className="reddit-own-badge">📌 Our post</span>}
                      <span className={`reddit-signal-pill reddit-signal-${post.signal}`}>{prettyLabel(post.signal)}</span>
                      <span className="reddit-post-time">{formatRelativeTime(post.created_at)}</span>
                    </div>
                  </div>
                  <strong className="reddit-post-title">{post.title}</strong>
                  <p className="reddit-post-excerpt">
                    {stripMd(post.excerpt)}
                  </p>
                  <div className="reddit-post-footer">
                    <div className="reddit-post-meta">
                      <span>u/{post.author}</span>
                      <span>▲ {post.score}</span>
                      <span>💬 {post.num_comments}</span>
                    </div>
                    <div className="reddit-post-actions">
                      <a
                        href={post.permalink}
                        target="_blank"
                        rel="noreferrer"
                        className="reddit-action-link reddit-action-link-primary"
                      >
                        Open ↗
                      </a>
                      <button
                        type="button"
                        className="reddit-action-link reddit-action-button"
                        onClick={() => onImportPost(post)}
                        disabled={pendingKey === `reddit-${post.id}`}
                      >
                        {pendingKey === `reddit-${post.id}` ? "…" : "Intake"}
                      </button>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </article>
    </aside>
  );
})
