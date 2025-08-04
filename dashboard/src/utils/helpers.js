export const getStatusColor = (status) => {
  switch (status) {
    case 'UP': return '#2563eb';
    case 'DOWN': return '#6b7280';
    case 'STARTING': return '#f59e0b';
    default: return '#9ca3af';
  }
};

export const getUsageColor = (usage) => {
  if (usage < 30) return '#2563eb';
  if (usage < 70) return '#3b82f6';
  return '#1d4ed8';
};

