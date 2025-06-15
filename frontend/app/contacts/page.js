import { FaEnvelope, FaPhone, FaLinkedin, FaGithub } from 'react-icons/fa';

export default function ContactsPage() {
  return (
    <div className="main-container">
      <div className="contacts-main-container">
        <h1 className="contacts-title">Contact Information</h1>
        
        <div className="contacts-main-card">
          <p className="contacts-intro">
            Feel free to reach out if you have any questions, suggestions, or would like to discuss this project further.
          </p>
          
          <div className="contacts-grid">
            <div className="contact-item-enhanced">
              <div className="contact-icon-enhanced">
                <FaEnvelope className="text-white" size={20} />
              </div>
              <div className="contact-content">
                <h3 className="contact-label">Email</h3>
                <div className="contact-value">
                  <a href="mailto:bernardomloguterres@gmail.com">
                    bernardomloguterres@gmail.com
                  </a>
                </div>
              </div>
            </div>
            
            <div className="contact-item-enhanced">
              <div className="contact-icon-enhanced">
                <FaPhone className="text-white" size={20} />
              </div>
              <div className="contact-content">
                <h3 className="contact-label">Phone</h3>
                <div className="contact-value">
                  <a href="tel:+351969019152">
                    +351 969 019 152
                  </a>
                </div>
              </div>
            </div>
            
            <div className="contact-item-enhanced">
              <div className="contact-icon-enhanced">
                <FaLinkedin className="text-white" size={20} />
              </div>
              <div className="contact-content">
                <h3 className="contact-label">LinkedIn</h3>
                <div className="contact-value">
                  <a 
                    href="https://www.linkedin.com/in/bernardoguterres/" 
                    target="_blank" 
                    rel="noopener noreferrer"
                  >
                    linkedin.com/in/bernardoguterres
                  </a>
                </div>
              </div>
            </div>
            
            <div className="contact-item-enhanced">
              <div className="contact-icon-enhanced">
                <FaGithub className="text-white" size={20} />
              </div>
              <div className="contact-content">
                <h3 className="contact-label">GitHub</h3>
                <div className="contact-value">
                  <a 
                    href="https://github.com/Martinnnnnns" 
                    target="_blank" 
                    rel="noopener noreferrer"
                  >
                    github.com/Martinnnnnns
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}