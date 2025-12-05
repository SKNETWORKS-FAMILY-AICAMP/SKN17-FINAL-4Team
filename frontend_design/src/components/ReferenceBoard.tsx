import React, { useState, useEffect, useCallback } from 'react';
import { Download, Lamp, Loader2, AlertCircle } from 'lucide-react';
import type { Page } from '../types/navigation';
import { api } from '../lib/api';

interface ReferenceBoardProps {
  onNavigate: (page: Page) => void;
  isLoggedIn: boolean;
  userPreferences?: any;
  onLogout?: () => void;
  userEmail?: string;
}

const styleCategories = [
  'vintage', 'luxury', 'natural', 'scandinavian', 'french',
  'lovely', 'pastel', 'modern', 'bohemian', 'classic',
  'industrial', 'minimal'
];

const categoryImages: { [key: string]: string[] } = {
  vintage: [
    'https://images.unsplash.com/photo-1725711362462-a0333461e1df?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx2aW50YWdlJTIwaW50ZXJpb3IlMjBkZXNpZ258ZW58MXx8fHwxNzY0MTM2MTU2fDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1710082777338-dcb6189ae64f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx2aW50YWdlJTIwaW50ZXJpb3IlMjByb29tfGVufDF8fHx8MTc2NDEzNDMwNnww&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1725711362462-a0333461e1df?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx2aW50YWdlJTIwaW50ZXJpb3IlMjBkZXNpZ258ZW58MXx8fHwxNzY0MTM2MTU2fDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1710082777338-dcb6189ae64f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx2aW50YWdlJTIwaW50ZXJpb3IlMjByb29tfGVufDF8fHx8MTc2NDEzNDMwNnww&ixlib=rb-4.1.0&q=80&w=1080'
  ],
  luxury: [
    'https://images.unsplash.com/photo-1687180498602-5a1046defaa4?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsdXh1cnklMjBpbnRlcmlvciUyMHJvb218ZW58MXx8fHwxNzY0MTM2MTU2fDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1687180498602-5a1046defaa4?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsdXh1cnklMjBpbnRlcmlvciUyMHJvb218ZW58MXx8fHwxNzY0MTM2MTU2fDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1687180498602-5a1046defaa4?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsdXh1cnklMjBpbnRlcmlvciUyMHJvb218ZW58MXx8fHwxNzY0MTM2MTU2fDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1687180498602-5a1046defaa4?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsdXh1cnklMjBpbnRlcmlvciUyMHJvb218ZW58MXx8fHwxNzY0MTM2MTU2fDA&ixlib=rb-4.1.0&q=80&w=1080'
  ],
  natural: [
    'https://images.unsplash.com/photo-1597562965673-42cc92e8408f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxuYXR1cmFsJTIwaW50ZXJpb3IlMjBzcGFjZXxlbnwxfHx8fDE3NjQxMzYxNTZ8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1597562965673-42cc92e8408f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxuYXR1cmFsJTIwaW50ZXJpb3IlMjBzcGFjZXxlbnwxfHx8fDE3NjQxMzYxNTZ8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1597562965673-42cc92e8408f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxuYXR1cmFsJTIwaW50ZXJpb3IlMjBzcGFjZXxlbnwxfHx8fDE3NjQxMzYxNTZ8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1597562965673-42cc92e8408f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxuYXR1cmFsJTIwaW50ZXJpb3IlMjBzcGFjZXxlbnwxfHx8fDE3NjQxMzYxNTZ8MA&ixlib=rb-4.1.0&q=80&w=1080'
  ],
  scandinavian: [
    'https://images.unsplash.com/photo-1724582586413-6b69e1c94a17?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzY2FuZGluYXZpYW4lMjBpbnRlcmlvcnxlbnwxfHx8fDE3NjQxMjUzODd8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1724582586413-6b69e1c94a17?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzY2FuZGluYXZpYW4lMjBpbnRlcmlvcnxlbnwxfHx8fDE3NjQxMjUzODd8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1724582586413-6b69e1c94a17?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzY2FuZGluYXZpYW4lMjBpbnRlcmlvcnxlbnwxfHx8fDE3NjQxMjUzODd8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1724582586413-6b69e1c94a17?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzY2FuZGluYXZpYW4lMjBpbnRlcmlvcnxlbnwxfHx8fDE3NjQxMjUzODd8MA&ixlib=rb-4.1.0&q=80&w=1080'
  ],
  french: [
    'https://images.unsplash.com/photo-1678775970375-05bbabcc6bcf?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmcmVuY2glMjBpbnRlcmlvciUyMGRlc2lnbnxlbnwxfHx8fDE3NjQxMzYxNTd8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1678775970375-05bbabcc6bcf?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmcmVuY2glMjBpbnRlcmlvciUyMGRlc2lnbnxlbnwxfHx8fDE3NjQxMzYxNTd8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1678775970375-05bbabcc6bcf?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmcmVuY2glMjBpbnRlcmlvciUyMGRlc2lnbnxlbnwxfHx8fDE3NjQxMzYxNTd8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1678775970375-05bbabcc6bcf?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmcmVuY2glMjBpbnRlcmlvciUyMGRlc2lnbnxlbnwxfHx8fDE3NjQxMzYxNTd8MA&ixlib=rb-4.1.0&q=80&w=1080'
  ],
  lovely: [
    'https://images.unsplash.com/photo-1756317058150-63264dea336c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsb3ZlbHklMjBjdXRlJTIwaW50ZXJpb3J8ZW58MXx8fHwxNzY0MTM2MTU3fDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1756317058150-63264dea336c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsb3ZlbHklMjBjdXRlJTIwaW50ZXJpb3J8ZW58MXx8fHwxNzY0MTM2MTU3fDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1756317058150-63264dea336c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsb3ZlbHklMjBjdXRlJTIwaW50ZXJpb3J8ZW58MXx8fHwxNzY0MTM2MTU3fDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1756317058150-63264dea336c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsb3ZlbHklMjBjdXRlJTIwaW50ZXJpb3J8ZW58MXx8fHwxNzY0MTM2MTU3fDA&ixlib=rb-4.1.0&q=80&w=1080'
  ],
  pastel: [
    'https://images.unsplash.com/photo-1632999101501-47bd016f7e46?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwYXN0ZWwlMjBpbnRlcmlvciUyMHJvb218ZW58MXx8fHwxNzY0MTM2MTU3fDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1632999101501-47bd016f7e46?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwYXN0ZWwlMjBpbnRlcmlvciUyMHJvb218ZW58MXx8fHwxNzY0MTM2MTU3fDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1632999101501-47bd016f7e46?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwYXN0ZWwlMjBpbnRlcmlvciUyMHJvb218ZW58MXx8fHwxNzY0MTM2MTU3fDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1632999101501-47bd016f7e46?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwYXN0ZWwlMjBpbnRlcmlvciUyMHJvb218ZW58MXx8fHwxNzY0MTM2MTU3fDA&ixlib=rb-4.1.0&q=80&w=1080'
  ],
  modern: [
    'https://images.unsplash.com/photo-1520106392146-ef585c111254?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBpbnRlcmlvciUyMGFwYXJ0bWVudHxlbnwxfHx8fDE3NjQxMzYxNTh8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1592401526914-7e5d94a8d6fa?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBpbnRlcmlvciUyMGxpdmluZyUyMHJvb218ZW58MXx8fHwxNzY0MDY2NDMzfDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1520106392146-ef585c111254?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBpbnRlcmlvciUyMGFwYXJ0bWVudHxlbnwxfHx8fDE3NjQxMzYxNTh8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1592401526914-7e5d94a8d6fa?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBpbnRlcmlvciUyMGxpdmluZyUyMHJvb218ZW58MXx8fHwxNzY0MDY2NDMzfDA&ixlib=rb-4.1.0&q=80&w=1080'
  ],
  bohemian: [
    'https://images.unsplash.com/photo-1600493504591-aa1849716b36?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxib2hlbWlhbiUyMGludGVyaW9yJTIwZGVzaWdufGVufDF8fHx8MTc2NDEzNjE1OHww&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1600493504591-aa1849716b36?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxib2hlbWlhbiUyMGludGVyaW9yJTIwZGVzaWdufGVufDF8fHx8MTc2NDEzNjE1OHww&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1600493504591-aa1849716b36?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxib2hlbWlhbiUyMGludGVyaW9yJTIwZGVzaWdufGVufDF8fHx8MTc2NDEzNjE1OHww&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1600493504591-aa1849716b36?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxib2hlbWlhbiUyMGludGVyaW9yJTIwZGVzaWdufGVufDF8fHx8MTc2NDEzNjE1OHww&ixlib=rb-4.1.0&q=80&w=1080'
  ],
  classic: [
    'https://images.unsplash.com/photo-1716058845923-9212b7e0887b?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjbGFzc2ljJTIwaW50ZXJpb3IlMjByb29tfGVufDF8fHx8MTc2NDEzNjE1OXww&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1716058845923-9212b7e0887b?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjbGFzc2ljJTIwaW50ZXJpb3IlMjByb29tfGVufDF8fHx8MTc2NDEzNjE1OXww&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1716058845923-9212b7e0887b?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjbGFzc2ljJTIwaW50ZXJpb3IlMjByb29tfGVufDF8fHx8MTc2NDEzNjE1OXww&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1716058845923-9212b7e0887b?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjbGFzc2ljJTIwaW50ZXJpb3IlMjByb29tfGVufDF8fHx8MTc2NDEzNjE1OXww&ixlib=rb-4.1.0&q=80&w=1080'
  ],
  industrial: [
    'https://images.unsplash.com/photo-1652716279221-439c33c3b835?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbmR1c3RyaWFsJTIwaW50ZXJpb3IlMjBsb2Z0fGVufDF8fHx8MTc2NDEzNjE1OXww&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1652716279221-439c33c3b835?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbmR1c3RyaWFsJTIwaW50ZXJpb3IlMjBsb2Z0fGVufDF8fHx8MTc2NDEzNjE1OXww&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1652716279221-439c33c3b835?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbmR1c3RyaWFsJTIwaW50ZXJpb3IlMjBsb2Z0fGVufDF8fHx8MTc2NDEzNjE1OXww&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1652716279221-439c33c3b835?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbmR1c3RyaWFsJTIwaW50ZXJpb3IlMjBsb2Z0fGVufDF8fHx8MTc2NDEzNjE1OXww&ixlib=rb-4.1.0&q=80&w=1080'
  ],
  minimal: [
    'https://images.unsplash.com/photo-1621363183028-c97aec91a9f3?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtaW5pbWFsJTIwaW50ZXJpb3IlMjB3aGl0ZXxlbnwxfHx8fDE3NjQxMzYxNTl8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1705321963943-de94bb3f0dd3?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtaW5pbWFsaXN0JTIwaW50ZXJpb3IlMjBkZXNpZ258ZW58MXx8fHwxNzY0MDczNjM0fDA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1621363183028-c97aec91a9f3?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtaW5pbWFsJTIwaW50ZXJpb3IlMjB3aGl0ZXxlbnwxfHx8fDE3NjQxMzYxNTl8MA&ixlib=rb-4.1.0&q=80&w=1080',
    'https://images.unsplash.com/photo-1705321963943-de94bb3f0dd3?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtaW5pbWFsaXN0JTIwaW50ZXJpb3IlMjBkZXNpZ258ZW58MXx8fHwxNzY0MDczNjM0fDA&ixlib=rb-4.1.0&q=80&w=1080'
  ]
};

export function ReferenceBoard({ onNavigate, isLoggedIn, userPreferences, onLogout, userEmail }: ReferenceBoardProps) {
  const [selectedStyles, setSelectedStyles] = useState<string[]>(
    userPreferences?.preferred_moods || userPreferences?.styles || []
  );
  const [loadingPrefs, setLoadingPrefs] = useState(false);
  const [error, setError] = useState('');

  // Scroll to top on mount
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  useEffect(() => {
    const moods = userPreferences?.preferred_moods || userPreferences?.styles || [];
    if (moods.length) {
      setSelectedStyles(moods);
    }
  }, [userPreferences]);

  const fetchPreferences = useCallback(async () => {
    setLoadingPrefs(true);
    setError('');
    try {
      const res = await api.get('/favorites/preferences/');
      setSelectedStyles(res.data?.preferred_moods ?? []);
    } catch (err) {
      console.error('선호도 정보를 불러오지 못했습니다.', err);
      setError('선호도 정보를 불러오지 못했습니다. 기본 이미지를 표시합니다.');
    } finally {
      setLoadingPrefs(false);
    }
  }, []);

  useEffect(() => {
    if (!userPreferences) {
      fetchPreferences();
    }
  }, [userPreferences, fetchPreferences]);

  const toggleStyle = (style: string) => {
    if (selectedStyles.includes(style)) {
      setSelectedStyles(selectedStyles.filter(s => s !== style));
    } else {
      setSelectedStyles([...selectedStyles, style]);
    }
  };

  const handleDownloadImage = async (imageUrl: string, index: number) => {
    try {
      // Fetch the image
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      
      // Create a temporary URL for the blob
      const blobUrl = window.URL.createObjectURL(blob);
      
      // Create a temporary anchor element and trigger download
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = `mood-on-reference-${index + 1}.jpg`;
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error('Download failed:', error);
      alert('이미지 다운로드에 실패했습니다.');
    }
  };

  // Get images based on selected styles
  const getDisplayImages = () => {
    if (selectedStyles.length === 0) {
      const allImages: string[] = [];
      styleCategories.forEach(style => {
        if (categoryImages[style]) {
          allImages.push(...categoryImages[style]);
        }
      });
      // Shuffle and return random images
      return allImages.sort(() => Math.random() - 0.5);
    }
    
    const allImages: string[] = [];
    selectedStyles.forEach(style => {
      if (categoryImages[style]) {
        allImages.push(...categoryImages[style]);
      }
    });
    
    return allImages;
  };

  const displayImages = getDisplayImages();
  const hasPreferences = selectedStyles.length > 0;

  return (
    <div className="min-h-screen bg-white overflow-y-auto">
      {/* Custom Header */}
      <div className="border-b border-blue-100 px-5 py-3.5 flex items-center justify-between bg-white/80 backdrop-blur-sm shadow-sm fixed top-0 left-0 right-0 z-50">
        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate('chat')}
            className="flex items-center gap-2.5 hover:opacity-80 transition-opacity"
          >
            <div className="w-9 h-9 bg-gradient-to-br from-blue-400 to-blue-300 rounded-full flex items-center justify-center shadow-md">
              <Lamp size={18} className="text-white" />
            </div>
            <span className="text-[20px] font-medium leading-none bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent select-none">
              MOOD ON
            </span>
          </button>
        </div>
        
        {/* Navigation Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onNavigate('mypage')}
            className="px-4 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all leading-none"
          >
            마이페이지
          </button>
          <button
            onClick={() => onNavigate('reference')}
            className="px-4 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all leading-none"
          >
            레퍼런스 보드
          </button>
          <button
            onClick={() => onNavigate('preference')}
            className="px-4 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all leading-none"
          >
            취향분석
          </button>
          <button
            onClick={onLogout}
            className="px-4 py-2 text-[15px] font-normal text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all leading-none"
          >
            로그아웃
          </button>
        </div>
      </div>
      
      <main className="pt-26 pb-16">
        <div className="max-w-7xl mx-auto px-6">
          {/* Top Section: Style Tags (Left) and Title/Description (Right) */}
          <div className="flex gap-8 mb-6">
            {/* Left: Style Tags (3 columns grid) */}
            <div className="flex-shrink-0">
              <div className="grid grid-cols-3 gap-3">
                {styleCategories.map((style) => (
                  <button
                    key={style}
                    onClick={() => toggleStyle(style)}
                    className={`px-6 py-2 rounded-full border-2 transition-all whitespace-nowrap ${
                      selectedStyles.includes(style)
                        ? 'bg-gradient-to-r from-pink-400 via-purple-400 to-blue-400 text-white border-transparent shadow-lg'
                        : 'bg-white text-purple-600 border-purple-300 hover:border-purple-400'
                    }`}
                  >
                    {style}
                  </button>
                ))}
              </div>
            </div>
            
            {/* Right: Title and Description */}
            <div className="flex-1">
              <h1 className="text-4xl mb-4">Reference Board</h1>
              <p className="text-gray-600 mb-2">
                원하는 무드 이미지가 없다면, 이제 손쉽게 MOOD ON의 레퍼런스 보드에서 찾아보세요.
              </p>
              <p className="text-gray-600 mb-2">
                빈티지부터 미니멀까지, 12개의 인테리어 카테고리로 분류된 이미지 레퍼런스를 찾아 저장하고, 당신만의 감성 취향을 찾아보는 건 어떨까요?
              </p>
              <p className="text-gray-600 mb-2">
                좌측에 있는 무드 태그를 클릭해서 원하는 감성 이미지를 찾아보세요. 마음에 들었다면 다운로드 버튼을 눌러서 이미지를 저장해볼 수 있습니다.
              </p>
            </div>
          </div>
          
          {/* Preference Survey Banner - Full Width */}
          <div className="relative h-20 rounded-3xl overflow-hidden mb-8 bg-gradient-to-r from-pink-100 to-purple-100 flex items-center justify-between px-8">
            <div className="text-xl">선호도가 바뀌셨나요? 언제든 당신의 선호도를 업데이트해주세요!</div>
            <button 
              onClick={() => onNavigate('preference')}
              className="px-6 py-2.5 bg-white rounded-full hover:bg-gray-50 transition-colors border border-gray-200"
            >
              선호도 업데이트 &gt;
            </button>
          </div>
          
          {/* Preference Message */}
          {hasPreferences && (
            <div className="mb-6">
              <p className="text-xl text-gray-700">
                {userEmail || (userPreferences?.gender ? `${userPreferences.gender} 회원` : '회원')}님이 선호하는 인테리어 무드 이미지입니다
              </p>
            </div>
          )}

          {loadingPrefs && (
            <div className="mb-4 flex items-center gap-2 text-sm text-gray-500">
              <Loader2 size={18} className="animate-spin" />
              선호 무드를 불러오는 중입니다...
            </div>
          )}

          {error && (
            <div className="mb-4 flex items-center gap-2 rounded-2xl border border-yellow-200 bg-yellow-50 px-4 py-3 text-sm text-yellow-800">
              <AlertCircle size={18} />
              {error}
            </div>
          )}
          
          {/* Image Grid */}
          {displayImages.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {displayImages.map((image, index) => (
                <div key={index} className="relative group">
                  <div className="aspect-square rounded-3xl overflow-hidden">
                    <img 
                      src={image}
                      alt={`Interior ${index + 1}`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <button 
                    onClick={() => handleDownloadImage(image, index)}
                    className="absolute bottom-4 left-1/2 -translate-x-1/2 px-6 py-2 bg-white rounded-full border border-gray-300 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-2 hover:bg-gray-50"
                  >
                    <Download size={16} />
                    download
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-16 text-gray-500">
              스타일을 선택하면 해당 이미지가 표시됩니다.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}