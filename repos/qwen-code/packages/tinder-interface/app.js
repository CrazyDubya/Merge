const characters = [
  {
    id: 'lyra',
    name: 'Lyra Vega',
    title: 'Galactic Cartographer',
    tagline: 'Mapping wormholes & midnight playlists',
    avatar:
      'https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=620&q=80',
    bio:
      'Navigator for the Interstellar Atlas Guild. I chart the shortcuts between galaxies, then soundtrack the journey with glitchy synthwave.',
    tags: ['Stargazer', 'Synthwave DJ', 'Chaos Theorist'],
    conversationStarters: [
      'So… are you more of a nebula sprint or black hole dive kind of explorer?',
      'I just mapped a shortcut through an aurora storm. Want to see the timelapse?',
    ],
    openings: [
      'I saved a seat by the observation deck. The Andromeda lights are absurd tonight.',
      'Question: if you could rename a galaxy, what would you call it? I have a few contenders.',
    ],
  },
  {
    id: 'atlas',
    name: 'Atlas Calder',
    title: 'Sentient Architect',
    tagline: 'Designing adaptive cities that breathe with their citizens',
    avatar:
      'https://images.unsplash.com/photo-1502823403499-6ccfcf4fb453?auto=format&fit=crop&w=620&q=80',
    bio:
      'I sculpt responsive megastructures that reconfigure in realtime. Lover of brutalism, bio-luminescent corridors, and excellent espresso.',
    tags: ['Urban Alchemist', 'Analog Film', 'Adaptive AI'],
    conversationStarters: [
      'Today I taught a skyscraper to sway with the wind like bamboo. Should I name it after you?',
      'Quick: pick a color palette for a sunrise market. I will sketch while you speak.',
    ],
    openings: [
      'If architecture is frozen music, this week I composed a jazz skyline.',
      'My drafting table is covered in notes about pocket parks. Come help me prioritize.',
    ],
  },
  {
    id: 'noor',
    name: 'Noor Elyth',
    title: 'Chrono-DJ & Memory Weaver',
    tagline: 'Mixing timelines to keep the party one step ahead',
    avatar:
      'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=620&q=80',
    bio:
      'I host rooftop raves for time travelers. I splice memories into beats so every drop feels like déjà vu and possibility at once.',
    tags: ['Temporal Beats', 'Aurora Collector', 'Zero-Gravity Dancer'],
    conversationStarters: [
      'I just looped a beat that syncs to your heartbeat. Want to try the remix?',
      'Describe your favorite memory and I will sample it into tonight’s set.',
    ],
    openings: [
      'Midnight is elastic tonight. Meet me between minutes?',
      'You look like someone who knows how to bend time without breaking it.',
    ],
  },
  {
    id: 'soren',
    name: 'Soren Hale',
    title: 'Aerial Botanist',
    tagline: 'Growing bioluminescent forests in the sky',
    avatar:
      'https://images.unsplash.com/photo-1544723795-3fb6469f5b39?auto=format&fit=crop&w=620&q=80',
    bio:
      'Caretaker of floating gardens. I graft constellations onto vines and let hummingbirds conduct the pollen ballet.',
    tags: ['Sky Gardens', 'Mycelium Networks', 'Sound Bath Enthusiast'],
    conversationStarters: [
      'I transplanted stardust orchids into a hover dome. They keep asking about you.',
      'What song would you play for a seed before it sprouts?',
    ],
    openings: [
      'Bring a mug. I brewed comet-tail tea to watch the seedlings wake.',
      'My greenhouse drones are making mischief again. Care to help discipline them?',
    ],
  },
];

const cardStack = document.getElementById('cardStack');
const swipeStatus = document.getElementById('swipeStatus');
const chatPanel = document.getElementById('chatPanel');
const chatAvatar = document.getElementById('chatAvatar');
const chatName = document.getElementById('chatName');
const chatTitle = document.getElementById('chatTitle');
const chatTimeline = document.getElementById('chatTimeline');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const chatBack = document.getElementById('chatBack');
const likeButton = document.getElementById('likeButton');
const dismissButton = document.getElementById('dismissButton');
const themeToggle = document.getElementById('themeToggle');

const cardTemplate = document.getElementById('cardTemplate');
const chatMessageTemplate = document.getElementById('chatMessageTemplate');

const cardCharacterMap = new WeakMap();
let activeIndex = 0;
let activeChatCharacter = null;

function renderStack() {
  cardStack.innerHTML = '';

  if (activeIndex >= characters.length) {
    renderEmptyState();
    updateActions();
    return;
  }

  const fragment = document.createDocumentFragment();
  const visible = characters.slice(activeIndex, activeIndex + 3);

  visible.forEach((character, index) => {
    const card = createCard(character, index);
    fragment.appendChild(card);
  });

  cardStack.appendChild(fragment);
  requestAnimationFrame(() => updateActions());
}

function renderEmptyState() {
  const empty = document.createElement('div');
  empty.className = 'deck__empty';
  empty.innerHTML = `
    <div class="deck__emptyBadge">All caught up ✨</div>
    <h2>Out of cards</h2>
    <p>Everyone has been introduced. Circle back soon for new cosmic encounters.</p>
    <button type="button" class="deck__reset" aria-label="Restart deck">Restart deck</button>
  `;
  empty.querySelector('.deck__reset').addEventListener('click', () => {
    activeIndex = 0;
    renderStack();
  });
  cardStack.appendChild(empty);
}

function createCard(character, offset) {
  const card = cardTemplate.content.firstElementChild.cloneNode(true);
  card.dataset.characterId = character.id;
  card.dataset.offset = String(offset);
  card.dataset.dragging = 'false';
  card.style.zIndex = String(100 - offset);
  cardCharacterMap.set(card, character);

  const body = card.querySelector('.card__body');
  const avatar = card.querySelector('.card__avatar');
  const name = card.querySelector('.card__name');
  const title = card.querySelector('.card__title');
  const bio = card.querySelector('.card__bio');
  const tags = card.querySelector('.card__tags');

  avatar.src = character.avatar;
  avatar.alt = `${character.name}'s portrait`;
  name.textContent = character.name;
  title.textContent = character.title;
  bio.textContent = character.bio;
  tags.innerHTML = '';
  character.tags.forEach((tag) => {
    const li = document.createElement('li');
    li.textContent = tag;
    tags.appendChild(li);
  });

  card.style.pointerEvents = offset === 0 ? 'auto' : 'none';
  attachSwipe(card, body, character);

  body.addEventListener('click', (event) => {
    if (card.dataset.dragging === 'true') {
      event.preventDefault();
      return;
    }
    openChat(character);
  });

  return card;
}

function attachSwipe(card, surface, character) {
  let pointerId = null;
  let startX = 0;
  let startY = 0;
  let dragging = false;

  function resetCard() {
    card.style.transition = 'transform 320ms ease';
    card.style.transform = '';
    card.classList.remove('is-like', 'is-dismiss');
    card.dataset.dragging = 'false';
  }

  surface.addEventListener('pointerdown', (event) => {
    if (card.dataset.offset !== '0') return;
    pointerId = event.pointerId;
    startX = event.clientX;
    startY = event.clientY;
    dragging = false;
    card.dataset.dragging = 'false';
    card.style.transition = 'none';
    surface.setPointerCapture(pointerId);
  });

  surface.addEventListener('pointermove', (event) => {
    if (pointerId !== event.pointerId) return;
    const dx = event.clientX - startX;
    const dy = event.clientY - startY;
    const distance = Math.hypot(dx, dy);
    if (!dragging && distance > 6) {
      dragging = true;
      card.dataset.dragging = 'true';
    }

    if (!dragging) return;

    const rotation = dx / 16;
    card.style.transform = `translate(${dx}px, ${dy}px) rotate(${rotation}deg)`;

    if (dx > 0) {
      card.classList.add('is-like');
      card.classList.remove('is-dismiss');
    } else if (dx < 0) {
      card.classList.add('is-dismiss');
      card.classList.remove('is-like');
    } else {
      card.classList.remove('is-like', 'is-dismiss');
    }
  });

  function handleRelease(event) {
    if (pointerId !== event.pointerId) return;
    surface.releasePointerCapture(pointerId);
    const dx = event.clientX - startX;
    const dy = event.clientY - startY;
    const velocityEnough = Math.abs(dx) > 120 || Math.abs(dx) > Math.abs(dy) * 1.6;

    if (dragging && velocityEnough) {
      finalizeSwipe(dx > 0 ? 'right' : 'left', card, character, { dx, dy });
    } else {
      if (!dragging) {
        resetCard();
        openChat(character);
      } else {
        resetCard();
      }
    }

    dragging = false;
    pointerId = null;
  }

  surface.addEventListener('pointerup', handleRelease);
  surface.addEventListener('pointercancel', handleRelease);
}

function finalizeSwipe(direction, card, character, delta = { dx: 0, dy: 0 }) {
  const travelX = direction === 'right' ? window.innerWidth * 0.9 : -window.innerWidth * 0.9;
  const travelY = delta.dy * 0.4;
  const rotation = direction === 'right' ? 28 : -28;

  card.classList.toggle('is-like', direction === 'right');
  card.classList.toggle('is-dismiss', direction === 'left');
  card.style.transition = 'transform 360ms ease-out, opacity 360ms ease-out';
  requestAnimationFrame(() => {
    card.style.transform = `translate(${travelX}px, ${travelY}px) rotate(${rotation}deg)`;
    card.style.opacity = '0';
  });

  card.addEventListener(
    'transitionend',
    () => {
      card.remove();
      activeIndex += 1;
      showSwipeStatus(direction, character);
      renderStack();
    },
    { once: true },
  );
}

function showSwipeStatus(direction, character) {
  const verb = direction === 'right' ? 'liked' : 'dismissed';
  swipeStatus.textContent = `${capitalize(character.name)} ${verb}.`;
  swipeStatus.classList.add('is-visible');
  setTimeout(() => {
    swipeStatus.classList.remove('is-visible');
  }, 1600);
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function triggerSwipe(direction) {
  const topCard = cardStack.querySelector('.card[data-offset="0"]');
  if (!topCard) return;
  const character = cardCharacterMap.get(topCard);
  if (!character) return;
  finalizeSwipe(direction, topCard, character, { dx: direction === 'right' ? 180 : -180, dy: 0 });
}

function updateActions() {
  const topCard = cardStack.querySelector('.card[data-offset="0"]');
  const hasCards = Boolean(topCard);
  likeButton.disabled = !hasCards;
  dismissButton.disabled = !hasCards;

  if (!hasCards) {
    likeButton.classList.add('is-disabled');
    dismissButton.classList.add('is-disabled');
  } else {
    likeButton.classList.remove('is-disabled');
    dismissButton.classList.remove('is-disabled');

    const remaining = cardStack.querySelectorAll('.card');
    remaining.forEach((card, index) => {
      card.dataset.offset = String(index);
      card.style.zIndex = String(100 - index);
      card.style.pointerEvents = index === 0 ? 'auto' : 'none';
      if (index > 0) {
        card.style.transform = '';
        card.classList.remove('is-like', 'is-dismiss');
      }
    });
  }
}

function openChat(character) {
  activeChatCharacter = character;
  chatPanel.hidden = false;
  chatAvatar.src = character.avatar;
  chatAvatar.alt = `${character.name}'s portrait`;
  chatName.textContent = character.name;
  chatTitle.textContent = `${character.title} • ${character.tagline}`;
  chatTimeline.innerHTML = '';

  character.openings.forEach((text) => {
    appendMessage({ text, author: 'them', timestamp: recentTimeOffset() });
  });

  const suggestion = character.conversationStarters[0];
  if (suggestion) {
    appendMessage({ text: suggestion, author: 'them', timestamp: recentTimeOffset(1) });
  }

  chatTimeline.scrollTop = chatTimeline.scrollHeight;
  chatInput.placeholder = `Message ${character.name}`;
  chatInput.focus({ preventScroll: true });
}

function recentTimeOffset(multiplier = 0) {
  const base = new Date();
  base.setMinutes(base.getMinutes() - 5 + multiplier);
  return base;
}

function appendMessage({ text, author, timestamp }) {
  const message = chatMessageTemplate.content.firstElementChild.cloneNode(true);
  const bubble = message.querySelector('.chat__bubble p');
  const timeEl = message.querySelector('time');

  bubble.textContent = text;
  message.classList.toggle('is-me', author === 'me');
  timeEl.textContent = formatTime(timestamp);
  chatTimeline.appendChild(message);
}

function formatTime(date) {
  return new Intl.DateTimeFormat([], {
    hour: 'numeric',
    minute: '2-digit',
  }).format(date);
}

chatForm.addEventListener('submit', (event) => {
  event.preventDefault();
  if (!activeChatCharacter) return;
  const text = chatInput.value.trim();
  if (!text) return;

  appendMessage({ text, author: 'me', timestamp: new Date() });
  chatInput.value = '';
  chatTimeline.scrollTop = chatTimeline.scrollHeight;

  setTimeout(() => {
    const response = generateResponse(activeChatCharacter);
    appendMessage({ text: response, author: 'them', timestamp: new Date() });
    chatTimeline.scrollTop = chatTimeline.scrollHeight;
  }, 700 + Math.random() * 900);
});

function generateResponse(character) {
  const options = [...character.conversationStarters.slice(1), ...character.openings];
  if (!options.length) {
    return `Tell me more about you. I want to tune this experience to your vibe.`;
  }
  const index = Math.floor(Math.random() * options.length);
  return options[index];
}

chatBack.addEventListener('click', () => {
  chatPanel.hidden = true;
  activeChatCharacter = null;
  chatTimeline.innerHTML = '';
});

likeButton.addEventListener('click', () => triggerSwipe('right'));
dismissButton.addEventListener('click', () => triggerSwipe('left'));

document.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowRight') {
    triggerSwipe('right');
  }
  if (event.key === 'ArrowLeft') {
    triggerSwipe('left');
  }
  if (event.key === 'Escape' && !chatPanel.hidden) {
    chatBack.click();
  }
});

function restoreThemePreference() {
  const saved = window.localStorage.getItem('swipeverse-theme');
  if (saved === 'dark') {
    document.documentElement.classList.add('dark');
  }
}

function toggleTheme() {
  document.documentElement.classList.toggle('dark');
  const mode = document.documentElement.classList.contains('dark') ? 'dark' : 'light';
  window.localStorage.setItem('swipeverse-theme', mode);
}

themeToggle.addEventListener('click', toggleTheme);
restoreThemePreference();
renderStack();
