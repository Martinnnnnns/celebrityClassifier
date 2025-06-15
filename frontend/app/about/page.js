import Image from 'next/image';

export default function AboutPage() {
  return (
    <div className="main-container">
      <div className="about-main-container">
        <h1 className="about-title">About Me</h1>
        
        <div className="about-main-card">
          <div className="about-profile-layout">
            {/* Profile Image Section */}
            <div className="about-image-section">
              <div className="about-profile-image">
                <Image 
                  src="/images/portrait.jpg" 
                  alt="Profile Picture" 
                  width={256}
                  height={256}
                  className="object-cover"
                  priority
                />
              </div>
            </div>
            
            {/* Bio Information Section */}
            <div className="about-bio-section">
              <h2 className="about-name">Bernardo Guterres</h2>
              
              <div className="about-bio-content">
                <p>
                  I'm currently pursuing a Bachelor's degree in Computer Science at King's College London. 
                  My academic journey is driven by a deep interest in Artificial Intelligence and Machine Learning, 
                  where I explore how intelligent systems can solve real-world problems and create impactful solutions.
                </p>

                <p>
                  I have hands-on experience working with languages like Python, Java, and SQL, and enjoy building projects 
                  that combine technical skills with creativity. I'm particularly drawn to Data Science, from extracting insights 
                  from data to training predictive models, and I'm eager to pursue a professional career in AI and Data Science.
                </p>

                <p>
                  Outside of the world of code, I'm an avid athlete with a passion for rock climbing and tennis. 
                  Balancing sports and studies keeps me focused, curious, and always pushing forward. 
                  I'm excited about contributing to the future of AI while continuing to grow as both a developer and a learner.
                </p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Skills and Education Sections */}
        <div className="about-additional-sections">
          <div className="about-section-card">
            <h2 className="about-section-title">Skills & Expertise</h2>
            <div className="about-skills-list">
              <div className="about-skill-item">
                <svg xmlns="http://www.w3.org/2000/svg" className="about-skill-icon" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Data Analysis & Visualization
              </div>
              <div className="about-skill-item">
                <svg xmlns="http://www.w3.org/2000/svg" className="about-skill-icon" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Machine Learning & AI
              </div>
              <div className="about-skill-item">
                <svg xmlns="http://www.w3.org/2000/svg" className="about-skill-icon" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Programming (Python, JavaScript, Java, SQL)
              </div>
              <div className="about-skill-item">
                <svg xmlns="http://www.w3.org/2000/svg" className="about-skill-icon" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Computer Vision
              </div>
              <div className="about-skill-item">
                <svg xmlns="http://www.w3.org/2000/svg" className="about-skill-icon" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Statistical Modeling
              </div>
            </div>
          </div>
          
          <div className="about-section-card">
            <h2 className="about-section-title">Education & Certifications</h2>
            <div className="about-education-list">
              <div className="about-education-item">
                <h3 className="about-education-title">BSc in Computer Science</h3>
                <p className="about-education-details">King's College London, 2023-2026</p>
              </div>
              <div className="about-education-item">
                <h3 className="about-education-title">Machine Learning Specialization</h3>
                <p className="about-education-details">DeepLearning.AI, 2024</p>
              </div>
              <div className="about-education-item">
                <h3 className="about-education-title">BSC - Data Science Job Simulation</h3>
                <p className="about-education-details">Forage, 2025</p>
              </div>
              <div className="about-education-item">
                <h3 className="about-education-title">JavaScript Coding</h3>
                <p className="about-education-details">iD Tech Camps, 2022</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}