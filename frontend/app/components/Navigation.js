'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navigation() {
  const pathname = usePathname();

  const navItems = [
    { href: '/', label: 'Classifier' },
    { href: '/about', label: 'About Me' },
    { href: '/purpose', label: 'Purpose' },
    { href: '/contacts', label: 'Contacts' },
    { href: '/other-work', label: 'Other Work' },
  ];

  return (
    <nav className="navbar-custom">
      <div className="navbar-container">
        <Link href="/" className="navbar-brand">
          Celebrity Classifier
        </Link>
        <ul className="nav-links">
          {navItems.map((item) => (
            <li key={item.href}>
              <Link
                href={item.href}
                className={`nav-link-custom ${pathname === item.href ? 'active' : ''}`}
              >
                {item.label}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}