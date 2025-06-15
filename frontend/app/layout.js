import './globals.css';
import Navigation from './components/Navigation.js';

export const metadata = {
  title: 'Celebrity Classifier',
  description: 'AI-powered sports celebrity image classification using machine learning',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen bg-gray-50">
          <Navigation />
          <main className="main-container">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}