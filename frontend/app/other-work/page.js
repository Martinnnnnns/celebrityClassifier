import Link from 'next/link';
import Image from 'next/image';

export default function OtherWorkPage() {
  return (
    <div className="main-container">
      <div className="otherwork-main-container">
        <h1 className="otherwork-title">Other Work</h1>
        
        <div className="otherwork-grid">
          {/* Project 1: COVID Data Explorer */}
          <div className="project-card">
            <div className="project-image-container">
              <Image 
                src="/images/coviddataexplorer.png" 
                alt="COVID Data Explorer"
                width={400}
                height={200}
                className="project-image"
                style={{ objectFit: 'cover' }}
              />
            </div>
            <div className="project-content">
              <h2 className="project-title">COVID Data Explorer</h2>
              <p className="project-description">
                A comprehensive GUI application for exploring COVID-19 data in London using sources from the GLA, 
                UK Government, and Google Mobility. Features multiple panels for dynamic data analysis and visualization 
                with interactive charts and statistical insights.
              </p>
              <div className="tech-tags-container">
                <span className="tech-tag-compact">Java</span>
                <span className="tech-tag-compact">Swing GUI</span>
                <span className="tech-tag-compact">Data Visualization</span>
                <span className="tech-tag-compact">API Integration</span>
              </div>
              <a 
                href="https://github.com/Martinnnnnns/CovidDataExplorer" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="project-link-enhanced"
              >
                View Project
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </a>
            </div>
          </div>
          
          {/* Project 2: University Society WebApp */}
          <div className="project-card">
            <div className="project-image-container">
              <Image 
                src="/images/societywebapp.png" 
                alt="University Society WebApp"
                width={400}
                height={200}
                className="project-image"
                style={{ objectFit: 'cover' }}
              />
            </div>
            <div className="project-content">
              <h2 className="project-title">University Society WebApp</h2>
              <p className="project-description">
                A comprehensive web platform for university societies to manage their online presence, events, 
                and memberships. Built with Flask backend and Next.js frontend - a complete full-stack web application 
                with user authentication and content management.
              </p>
              <div className="tech-tags-container">
                <span className="tech-tag-compact">Flask</span>
                <span className="tech-tag-compact">Next.js</span>
                <span className="tech-tag-compact">Full Stack</span>
                <span className="tech-tag-compact">PostgreSQL</span>
              </div>
              <a 
                href="https://kclsu.click/" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="project-link-enhanced"
              >
                View Project
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </a>
            </div>
          </div>

          {/* Project 3: Lisbon House Price Predictor */}
          <div className="project-card">
            <div className="project-image-container">
              <Image 
                src="/images/lisbonhousepricepredictor.png" 
                alt="Lisbon House Price Predictor"
                width={400}
                height={200}
                className="project-image"
                style={{ objectFit: 'cover' }}
              />
            </div>
            <div className="project-content">
              <h2 className="project-title">Lisbon House Price Predictor</h2>
              <p className="project-description">
                An advanced machine learning project that predicts house prices in Lisbon using various algorithms 
                including Linear Regression, Support Vector Machines, and Random Forests. Features interactive visualizations, 
                comprehensive model comparisons, and a modern web interface for exploring real estate predictions.
              </p>
              <div className="tech-tags-container">
                <span className="tech-tag-compact">Python</span>
                <span className="tech-tag-compact">Scikit-learn</span>
                <span className="tech-tag-compact">Pandas</span>
                <span className="tech-tag-compact">Next.js</span>
                <span className="tech-tag-compact">Machine Learning</span>
                <span className="tech-tag-compact">Data Visualization</span>
              </div>
              <a href="https://lisbonhousepricepredictor.vercel.app/"
                target="_blank"
                rel="noopener noreferrer"
                className="project-link-enhanced">
                View Project
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </a>
            </div>
          </div>

          {/* Project 4: Blank Template */}
          <div className="project-card">
            <div className="project-image-container">
              <span className="text-white text-xl font-bold">NBA MVP Predictor</span>
            </div>
            <div className="project-content">
              <h2 className="project-title">NBA MVP Predictor</h2>
              <p className="project-description">
                A machine learning project that predicts NBA MVP winners using historical player statistics, team performance, 
                and voting data scraped from Basketball Reference. Applies Ridge Regression and Random Forest algorithms 
                with advanced feature engineering and backtesting validation. Currently being developed with ongoing model 
                improvements and accuracy enhancements.
              </p>
              <div className="tech-tags-container">
                <span className="tech-tag-compact">Python</span>
                <span className="tech-tag-compact">Data Scraping</span>
                <span className="tech-tag-compact">Predictive Modeling</span>
                <span className="tech-tag-compact">Selenium</span>
              </div>
              <a href="#" className="project-link-enhanced">
                View Project
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}