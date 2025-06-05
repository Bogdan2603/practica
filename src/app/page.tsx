// app/page.tsx (NOUL fișier, la rădăcina folderului app)
import { redirect } from 'next/navigation';

export default function RootPage() {
  // Redirecționează permanent către /chat
  redirect('/chat'); 

  // Nu este nevoie de return JSX aici deoarece redirect() oprește execuția.
  // Dar, pentru conformitate, poți adăuga return null dacă dă vreo eroare de linting.
  // return null;
}