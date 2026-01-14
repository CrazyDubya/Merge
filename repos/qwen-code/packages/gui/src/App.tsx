import React from 'react';

const App = () => {
  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <div style={{ width: '200px', borderRight: '1px solid #ccc' }}>
        <h2>Sidebar</h2>
        <ul>
          <li>Chat</li>
          <li>Workspace</li>
          <li>Tools</li>
          <li>Settings</li>
        </ul>
      </div>
      <div style={{ flex: 1, padding: '20px' }}>
        <h1>Main Content</h1>
        <p>This is where the main content will go.</p>
      </div>
    </div>
  );
};

export default App;
