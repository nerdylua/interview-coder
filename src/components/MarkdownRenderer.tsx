import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        components={{
          // Custom code block renderer
          code({ className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';
            const isInline = !className;
            
            return !isInline && language ? (
              <SyntaxHighlighter
                style={dracula as any}
                language={language}
                PreTag="div"
                customStyle={{
                  margin: '0.5rem 0',
                  padding: '0.75rem',
                  fontSize: '12px',
                  backgroundColor: 'rgba(22, 27, 34, 0.5)',
                  borderRadius: '4px'
                }}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code 
                className="bg-black/30 px-1 py-0.5 rounded text-xs font-mono" 
                {...props}
              >
                {children}
              </code>
            );
          },
          // Custom paragraph renderer
          p({ children }) {
            return <div className="mb-2">{children}</div>;
          },
          // Custom list item renderer
          li({ children }) {
            return (
              <div className="flex items-start gap-2 mb-1">
                <div className="w-1 h-1 rounded-full bg-blue-400/80 mt-2 shrink-0" />
                <div className="flex-1">{children}</div>
              </div>
            );
          },
          // Custom unordered list renderer
          ul({ children }) {
            return <div className="space-y-1">{children}</div>;
          },
          // Custom ordered list renderer
          ol({ children }) {
            return <div className="space-y-1">{children}</div>;
          },
          // Custom heading renderers
          h1({ children }) {
            return <div className="font-bold text-white/90 text-sm mb-2 pb-1 border-b border-white/10">{children}</div>;
          },
          h2({ children }) {
            return <div className="font-semibold text-white/80 text-sm mt-3 mb-1">{children}</div>;
          },
          h3({ children }) {
            return <div className="font-medium text-white/70 text-sm mt-2 mb-1">{children}</div>;
          },
          // Custom strong/bold renderer
          strong({ children }) {
            return <span className="font-semibold text-white/90">{children}</span>;
          },
          // Custom emphasis/italic renderer
          em({ children }) {
            return <span className="italic text-white/80">{children}</span>;
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;